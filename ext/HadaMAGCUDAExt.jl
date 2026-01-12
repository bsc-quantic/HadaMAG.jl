module HadaMAGCUDAExt

using HadaMAG
using CUDA

# Workspace for SRE chunk computation on one GPU
# TODO: Do we really need mutable here?
mutable struct SREChunkWorkspace
    ψd::CuArray{ComplexF64,1} # device wavefunction
    X::CuArray{Float64,2} # (N, max_batch)
    masks_h::Vector{UInt64} # pinned host masks
    masks_d::CuArray{UInt64,1} # device masks (length max_batch)
    out_p2::CuArray{Float64,2} # partials (blocks_x, max_batch)
    out_mq::CuArray{Float64,2} # same shape, for m_q (or m4)
    threads::Int
end

function SREChunkWorkspace(ψ; max_batch::Int = 64, threads::Int = 256)
    # Materialize ψ once and upload once
    ψh = collect(ComplexF64, data(ψ))
    ψd = CuArray(ψh)
    N = length(ψh)

    # Work buffers
    X = CuArray{Float64}(undef, N, max_batch)
    masks_h = CUDA.pin(Vector{UInt64}(undef, max_batch))
    masks_d = CuArray{UInt64}(undef, max_batch)

    blocks_x = min(1024, cld(N, threads*2))
    out_p2 = CuArray{Float64}(undef, blocks_x, max_batch)
    out_mq = CuArray{Float64}(undef, blocks_x, max_batch)

    return SREChunkWorkspace(ψd, X, masks_h, masks_d, out_p2, out_mq, threads)
end

# global accumulators (device scalars)
mutable struct AccumGPU
    p2::CuArray{Float64,1} # length 1
    m4::CuArray{Float64,1} # length 1
end
AccumGPU() = AccumGPU(CUDA.zeros(Float64, 1), CUDA.zeros(Float64, 1))

# Fused first `fuse_stages` FWHT stages inside shared memory, per tile & column.
# x: (N,B) matrix. N must be power of two; pick TILE so TILE | N and TILE is a power of two.
# Each block processes (tile_id, col_id) = (blockIdx.x, blockIdx.y).
function k_fwht_head_batched!(x, ::Val{TILE}, ::Val{FUSE}) where {TILE,FUSE}
    N, B = size(x)
    col = blockIdx().y
    tile = blockIdx().x - 1 # 0-based tile id
    if col > B
        ;
        return;
    end
    base = tile * TILE
    if base >= N
        ;
        return;
    end

    # shared memory buffer for the tile
    smem = @cuDynamicSharedMem(eltype(x), TILE)

    # thread-strided load from global to shared
    tid = threadIdx().x
    nthreads = blockDim().x
    @inbounds for i = tid:nthreads:TILE
        smem[i] = x[base+i, col]
    end
    sync_threads()

    # do FUSE small-stride stages in shared mem
    @inbounds begin
        @assert (1 << FUSE) <= TILE
        for s = 0:(FUSE-1)
            step = 1 << s
            period = step << 1
            # map k∈[0, TILE/2) to (aidx,bidx) pairs
            for k0 = (tid-1):nthreads:((TILE>>>1)-1) # 0-based
                group = k0 ÷ step
                offs = k0 % step
                aidx = group * period + offs + 1 # 1-based
                bidx = aidx + step
                a = smem[aidx];
                b = smem[bidx]
                smem[aidx] = a + b
                smem[bidx] = a - b
            end
            sync_threads()
        end
    end

    # write back to global
    @inbounds for i = tid:nthreads:TILE
        x[base+i, col] = smem[i]
    end
    return
end

function fwht_head_batched!(
    x::CuArray{T,2};
    tile::Int = 1024,
    fuse_stages::Int = 10,
    threads::Int = 256,
    stream = nothing,
) where {T<:AbstractFloat}
    N, B = size(x);
    @assert ispow2(N)
    @assert ispow2(tile) && (N % tile == 0) "tile must be power of two and divide N"
    @assert (1<<fuse_stages) <= tile "fuse_stages too large for tile"
    tiles = N ÷ tile
    shmem = sizeof(T) * tile
    if stream === nothing
        @cuda threads=threads blocks=(tiles, B) shmem=shmem k_fwht_head_batched!(
            x,
            Val(tile),
            Val(fuse_stages),
        )
    else
        @cuda threads=threads blocks=(tiles, B) shmem=shmem stream=stream k_fwht_head_batched!(
            x,
            Val(tile),
            Val(fuse_stages),
        )
    end
    return x
end

function fwht_tail_batched!(
    x::CuArray{Float64,2};
    start_stage::Int = 10,
    threads::Int = 256,
    stream = nothing,
)
    N, B = size(x);
    @assert ispow2(N)
    L = trailing_zeros(N)
    blocks_x = min(cld(N>>>1, threads), 65_535)
    blocks_y = B
    if stream === nothing
        for s = Int32(start_stage):Int32(L-1)
            @cuda threads=threads blocks=(blocks_x, blocks_y) k_fwht_stage_batched!(
                x,
                Int32(1<<s),
            )
        end
    else
        for s = Int32(start_stage):Int32(L-1)
            @cuda threads=threads blocks=(blocks_x, blocks_y) stream=stream k_fwht_stage_batched!(
                x,
                Int32(1<<s),
            )
        end
    end
    x
end

# reduction kernel that atomically adds into d_p2[1], d_m4[1]
function k_reduce_accum!(x, d_p2, d_m4)
    tid = threadIdx().x
    n = length(x)
    g = blockDim().x * 2 * gridDim().x
    i = (blockIdx().x-1) * (blockDim().x*2) + tid

    # shared partials
    smem = @cuDynamicSharedMem(Float64, 2*blockDim().x)
    p2 = 0.0
    m = 0.0
    @inbounds while i <= n
        v1 = x[i];
        v1s = v1*v1;
        p2 += v1s;
        m += v1s*v1s
        j = i + blockDim().x
        if j <= n
            v2 = x[j];
            v2s = v2*v2;
            p2 += v2s;
            m += v2s*v2s
        end
        i += g
    end
    smem[tid] = p2
    smem[tid+blockDim().x] = m
    sync_threads()

    off = blockDim().x >>> 1
    while off > 0
        if tid <= off
            smem[tid] += smem[tid+off]
            smem[tid+blockDim().x] += smem[tid+blockDim().x+off]
        end
        sync_threads()
        off >>>= 1
    end

    if tid == 1
        # CC >= 6.0 supports atomic add for Float64
        CUDA.atomic_add!(pointer(d_p2), 1, smem[1])
        CUDA.atomic_add!(pointer(d_m4), 1, smem[1+blockDim().x])
    end
    return
end

function reduce_accum!(
    x::CuArray{Float64,1},
    acc::AccumGPU;
    threads = 256,
    stream = nothing,
)
    n = length(x)
    blocks = min(1024, cld(n, threads*2))
    shmem = sizeof(Float64) * threads * 2
    if stream === nothing
        @cuda threads=threads blocks=blocks shmem=shmem k_reduce_accum!(x, acc.p2, acc.m4)
    else
        @cuda threads=threads blocks=blocks shmem=shmem stream=stream k_reduce_accum!(
            x,
            acc.p2,
            acc.m4,
        )
    end
    return
end

# BUILD (batched)
function k_build_inVR_batched!(ψ, masks, out)  # out: (N,B)
    N = size(out, 1)
    B = size(out, 2)
    b = blockIdx().y
    if b > B
        ;
        return;
    end
    mask = masks[b] & UInt64(N-1)

    tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    i0 = tid - 1

    @inbounds for i = i0:step:(N-1)
        ii = Int(i + 1)
        idx = Int((UInt64(i) ⊻ mask) + 1)
        zi = ψ[ii];
        zj = ψ[idx]
        ri, ii_ = real(zi), imag(zi)
        rj, ij = real(zj), imag(zj)
        out[ii, b] = muladd(ri, rj + ij, ii_*(ij - rj))
    end
    return
end

function build_inVR_batched!(
    ψd::CuArray{ComplexF64,1},
    masks_d::CuArray{UInt64,1},
    out::CuArray{Float64,2};
    threads = 256,
    stream = nothing,
)
    N, B = size(out)
    blocks_x = min(cld(N, threads), 65_535)
    # blocks_y = size(masks_d,1)   # number of active columns
    blocks_y = B
    if stream === nothing
        @cuda threads=threads blocks=(blocks_x, blocks_y) k_build_inVR_batched!(
            ψd,
            masks_d,
            out,
        )
    else
        @cuda threads=threads blocks=(blocks_x, blocks_y) stream=stream k_build_inVR_batched!(
            ψd,
            masks_d,
            out,
        )
    end
    out
end

function k_fwht_stage_batched!(x, stride32::Int32)
    N = size(x, 1)
    b = blockIdx().y
    if b > size(x, 2)
        ;
        return;
    end
    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    half = N >>> 1
    s = Int(stride32)

    k0 = tid - 1
    @inbounds for k = k0:step:(half-1)
        group = k ÷ s
        offs = k % s
        base = group * (s << 1) + offs
        aidx = base + 1
        bidx = aidx + s
        a = x[aidx, b];
        c = x[bidx, b]
        x[aidx, b] = a + c
        x[bidx, b] = a - c
    end
    return
end
function fwht_batched!(x::CuArray{Float64,2}; threads = 256, stream = nothing)
    N, B = size(x);
    @assert ispow2(N)
    L = trailing_zeros(N)
    blocks_x = min(cld(N>>>1, threads), 65_535)
    blocks_y = B
    if stream === nothing
        for s = 0:Int32(L-1)
            @cuda threads=threads blocks=(blocks_x, blocks_y) k_fwht_stage_batched!(
                x,
                Int32(1<<s),
            )
        end
    else
        for s = 0:Int32(L-1)
            @cuda threads=threads blocks=(blocks_x, blocks_y) stream=stream k_fwht_stage_batched!(
                x,
                Int32(1<<s),
            )
        end
    end
    x
end

function k_reduce_batched_accum!(x, d_p2, d_m4)
    N, B = size(x)
    tid = threadIdx().x
    b = blockIdx().y
    if b > B
        ;
        return;
    end
    g = blockDim().x * 2 * gridDim().x
    i = (blockIdx().x-1) * (blockDim().x*2) + tid

    smem = @cuDynamicSharedMem(Float64, 2*blockDim().x)
    p2 = 0.0;
    m = 0.0
    @inbounds while i <= N
        v1 = x[i, b];
        v1s = v1*v1;
        p2 += v1s;
        m += v1s*v1s
        j = i + blockDim().x
        if j <= N
            v2 = x[j, b];
            v2s = v2*v2;
            p2 += v2s;
            m += v2s*v2s
        end
        i += g
    end
    smem[tid] = p2
    smem[tid+blockDim().x] = m
    sync_threads()

    off = blockDim().x >>> 1
    while off > 0
        if tid <= off
            smem[tid] += smem[tid+off]
            smem[tid+blockDim().x] += smem[tid+blockDim().x+off]
        end
        sync_threads()
        off >>>= 1
    end
    if tid == 1
        CUDA.atomic_add!(pointer(d_p2), 1, smem[1])
        CUDA.atomic_add!(pointer(d_m4), 1, smem[1+blockDim().x])
    end
    return
end

function reduce_batched_accum!(
    x::CuArray{Float64,2},
    acc::AccumGPU;
    threads = 256,
    stream = nothing,
)
    N, B = size(x)
    blocks_x = min(1024, cld(N, threads*2))
    blocks_y = B
    shmem = sizeof(Float64) * threads * 2
    if stream === nothing
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem k_reduce_batched_accum!(
            x,
            acc.p2,
            acc.m4,
        )
    else
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem stream=stream k_reduce_batched_accum!(
            x,
            acc.p2,
            acc.m4,
        )
    end
end
function k_reduce_cols_2moments!(X, out_p2, out_m4)
    N, B = size(X)
    b = blockIdx().y
    if b > B
        ;
        return;
    end

    tid = threadIdx().x
    nthr = blockDim().x
    g = nthr * 2 * gridDim().x
    i = (blockIdx().x-1) * (nthr*2) + tid

    # one shared buffer of length 4*nthr: [p2 | cp2 | m4 | cm4]
    smem = @cuDynamicSharedMem(Float64, 4*blockDim().x)
    off_cp2 = nthr
    off_m4 = 2*nthr
    off_cm4 = 3*nthr

    s_p2 = 0.0;
    c_p2 = 0.0
    s_m4 = 0.0;
    c_m4 = 0.0

    @inbounds while i <= N
        v1 = X[i, b];
        v1s = v1*v1
        # Kahan for p2
        y = v1s - c_p2;
        t = s_p2 + y;
        c_p2 = (t - s_p2) - y;
        s_p2 = t
        # Kahan for m4
        v14 = v1s*v1s
        y = v14 - c_m4;
        t = s_m4 + y;
        c_m4 = (t - s_m4) - y;
        s_m4 = t

        j = i + nthr
        if j <= N
            v2 = X[j, b];
            v2s = v2*v2
            y = v2s - c_p2;
            t = s_p2 + y;
            c_p2 = (t - s_p2) - y;
            s_p2 = t
            v24 = v2s*v2s
            y = v24 - c_m4;
            t = s_m4 + y;
            c_m4 = (t - s_m4) - y;
            s_m4 = t
        end
        i += g
    end

    # write thread partials to shared memory
    smem[tid] = s_p2
    smem[tid+off_cp2] = c_p2
    smem[tid+off_m4] = s_m4
    smem[tid+off_cm4] = c_m4
    sync_threads()

    # pairwise fold
    off = nthr >>> 1
    while off > 0
        if tid <= off
            smem[tid] += smem[tid+off]
            smem[tid+off_cp2] += smem[tid+off_cp2+off]
            smem[tid+off_m4] += smem[tid+off_m4+off]
            smem[tid+off_cm4] += smem[tid+off_cm4+off]
        end
        sync_threads()
        off >>>= 1
    end

    if tid == 1
        out_p2[blockIdx().x, b] = smem[1] + smem[1+off_cp2]
        out_m4[blockIdx().x, b] = smem[1+off_m4] + smem[1+off_cm4]
    end
    return
end

function k_fwht_last_stage_and_reduce_qv!(X, out_p2, out_mq, stride32::Int32, ::Val{2})
    N, B = size(X)
    b = blockIdx().y
    if b > B
        ;
        return;
    end

    tid = threadIdx().x
    nthr = blockDim().x
    step = nthr * gridDim().x
    half = N >>> 1
    s = Int(stride32)

    # Kahan accumulators for this thread
    s_p2 = 0.0;
    c_p2 = 0.0
    s_m4 = 0.0;
    c_m4 = 0.0

    k0 = (blockIdx().x-1)*nthr + tid - 1

    @inbounds for k = k0:step:(half-1)
        group = k ÷ s
        offs = k % s
        base = group * (s << 1) + offs
        aidx = base + 1
        bidx = aidx + s

        a = X[aidx, b]
        c = X[bidx, b]

        # last-stage butterfly
        y = a + c
        z = a - c
        X[aidx, b] = y
        X[bidx, b] = z

        # accumulate contributions to p2 and m4
        y2 = y*y
        z2 = z*z

        # p2
        for v2 in (y2, z2)
            yk = v2 - c_p2
            tk = s_p2 + yk
            c_p2 = (tk - s_p2) - yk
            s_p2 = tk
        end

        # m4
        y4 = y2*y2
        z4 = z2*z2
        for v4 in (y4, z4)
            yk = v4 - c_m4
            tk = s_m4 + yk
            c_m4 = (tk - s_m4) - yk
            s_m4 = tk
        end
    end

    # shared reduction
    smem = @cuDynamicSharedMem(Float64, 4*nthr)
    off_cp2 = nthr;
    off_m4 = 2*nthr;
    off_cm4 = 3*nthr

    smem[tid] = s_p2
    smem[tid+off_cp2] = c_p2
    smem[tid+off_m4] = s_m4
    smem[tid+off_cm4] = c_m4
    sync_threads()

    off = nthr >>> 1
    while off > 0
        if tid <= off
            smem[tid] += smem[tid+off]
            smem[tid+off_cp2] += smem[tid+off_cp2+off]
            smem[tid+off_m4] += smem[tid+off_m4+off]
            smem[tid+off_cm4] += smem[tid+off_cm4+off]
        end
        sync_threads()
        off >>>= 1
    end

    if tid == 1
        out_p2[blockIdx().x, b] = smem[1] + smem[1+off_cp2]
        out_mq[blockIdx().x, b] = smem[1+off_m4] + smem[1+off_cm4]
    end
    return
end


function k_fwht_last_stage_and_reduce_qv!(
    X,
    out_p2,
    out_mq,
    stride32::Int32,
    ::Val{Q},
) where {Q}
    N, B = size(X)
    b = blockIdx().y
    if b > B
        ;
        return;
    end

    tid = threadIdx().x
    nthr = blockDim().x
    step = nthr * gridDim().x
    half = N >>> 1
    s = Int(stride32)
    q = Float64(Q)

    # Kahan accumulators for this thread
    s_p2 = 0.0;
    c_p2 = 0.0
    s_mq = 0.0;
    c_mq = 0.0

    k0 = (blockIdx().x-1)*nthr + tid - 1

    @inbounds for k = k0:step:(half-1)
        group = k ÷ s
        offs = k % s
        base = group * (s << 1) + offs
        aidx = base + 1
        bidx = aidx + s

        a = X[aidx, b]
        c = X[bidx, b]

        # last-stage butterfly
        y = a + c
        z = a - c
        X[aidx, b] = y
        X[bidx, b] = z

        # accumulate contributions to p2 and mq
        y2 = y*y
        z2 = z*z

        # p2
        for v2 in (y2, z2)
            yk = v2 - c_p2
            tk = s_p2 + yk
            c_p2 = (tk - s_p2) - yk
            s_p2 = tk
        end

        # mq
        yq = y2^q
        zq = z2^q
        for vq in (yq, zq)
            yk = vq - c_mq
            tk = s_mq + yk
            c_mq = (tk - s_mq) - yk
            s_mq = tk
        end
    end

    # shared reduction
    smem = @cuDynamicSharedMem(Float64, 4*nthr)
    off_cp2 = nthr;
    off_mq = 2*nthr;
    off_cmq = 3*nthr

    smem[tid] = s_p2
    smem[tid+off_cp2] = c_p2
    smem[tid+off_mq] = s_mq
    smem[tid+off_cmq] = c_mq
    sync_threads()

    off = nthr >>> 1
    while off > 0
        if tid <= off
            smem[tid] += smem[tid+off]
            smem[tid+off_cp2] += smem[tid+off_cp2+off]
            smem[tid+off_mq] += smem[tid+off_mq+off]
            smem[tid+off_cmq] += smem[tid+off_cmq+off]
        end
        sync_threads()
        off >>>= 1
    end

    if tid == 1
        out_p2[blockIdx().x, b] = smem[1] + smem[1+off_cp2]
        out_mq[blockIdx().x, b] = smem[1+off_mq] + smem[1+off_cmq]
    end
    return
end

function fwht_tail_and_reduce_partials2!(
    X::CuArray{Float64,2},
    out_p2::CuArray{Float64,2},
    out_mq::CuArray{Float64,2},
    q::Val{Q};
    start_stage::Int,
    threads::Int = 256,
    stream = nothing,
) where {Q}
    N, B = size(X)
    @assert ispow2(N)
    L = trailing_zeros(N)
    @assert 0 ≤ start_stage < L
    @assert size(out_p2, 2) ≥ B
    @assert size(out_mq, 2) ≥ B

    blocks_x = size(out_p2, 1)
    blocks_y = B

    shmem = sizeof(Float64) * threads * 4

    # tail stages except last
    if start_stage ≤ L - 2
        for s = Int32(start_stage):Int32(L-2)
            if stream === nothing
                @cuda threads=threads blocks=(blocks_x, blocks_y) k_fwht_stage_batched!(
                    X,
                    Int32(1 << s),
                )
            else
                @cuda threads=threads blocks=(blocks_x, blocks_y) stream=stream k_fwht_stage_batched!(
                    X,
                    Int32(1 << s),
                )
            end
        end
    end

    last_stride = Int32(1 << (L - 1))
    if stream === nothing
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem k_fwht_last_stage_and_reduce_qv!(
            X,
            out_p2,
            out_mq,
            last_stride,
            q,
        )
    else
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem stream=stream k_fwht_last_stage_and_reduce_qv!(
            X,
            out_p2,
            out_mq,
            last_stride,
            q,
        )
    end

    return out_p2, out_mq
end

function fwht_tail_and_reduce_partials!(
    X::CuArray{Float64,2},
    q::Val{Q};
    start_stage::Int,
    threads::Int = 256,
    stream = nothing,
) where {Q}
    N, B = size(X)
    @assert ispow2(N)
    L = trailing_zeros(N)
    @assert 0 ≤ start_stage < L "invalid start_stage for FWHT tail"

    # FWHT tail grid: same as fwht_tail_batched!
    blocks_x = min(cld(N >>> 1, threads), 65_535)
    blocks_y = B

    # allocate partials for this call
    out_p2 = CuArray{Float64}(undef, blocks_x, B)
    out_mq = CuArray{Float64}(undef, blocks_x, B)

    # all tail stages except the last: plain FWHT stages
    if start_stage ≤ L - 2
        for s = Int32(start_stage):Int32(L-2)
            if stream === nothing
                @cuda threads=threads blocks=(blocks_x, blocks_y) k_fwht_stage_batched!(
                    X,
                    Int32(1 << s),
                )
            else
                @cuda threads=threads blocks=(blocks_x, blocks_y) stream=stream k_fwht_stage_batched!(
                    X,
                    Int32(1 << s),
                )
            end
        end
    end

    # last stage + reduction fused
    last_stride = Int32(1 << (L - 1))
    shmem = sizeof(Float64) * threads * 4

    if stream === nothing
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem k_fwht_last_stage_and_reduce_qv!(
            X,
            out_p2,
            out_mq,
            last_stride,
            q,
        )
    else
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem stream=stream k_fwht_last_stage_and_reduce_qv!(
            X,
            out_p2,
            out_mq,
            last_stride,
            q,
        )
    end

    return out_p2, out_mq
end

# convenience for integer q: forward to float-typed Val
function fwht_tail_and_reduce_partials!(
    X::CuArray{Float64,2},
    ::Val{Q};
    start_stage::Int,
    threads::Int = 256,
    stream = nothing,
) where {Q<:Integer}
    return fwht_tail_and_reduce_partials!(
        X,
        Val(Float64(Q));
        start_stage = start_stage,
        threads = threads,
        stream = stream,
    )
end

function k_reduce_cols_2moments_qv!(X, out_p2, out_mq, ::Val{Q}) where {Q<:AbstractFloat}
    N, B = size(X)
    b = blockIdx().y
    if b > B
        ;
        return;
    end

    tid = threadIdx().x;
    nthr = blockDim().x
    g = nthr * 2 * gridDim().x
    i = (blockIdx().x-1) * (nthr*2) + tid

    smem = @cuDynamicSharedMem(Float64, 4*blockDim().x)
    off_cp2 = nthr;
    off_mq = 2*nthr;
    off_cmq = 3*nthr

    s_p2 = 0.0;
    c_p2 = 0.0
    s_mq = 0.0;
    c_mq = 0.0
    q = Float64(Q) # compile-time constant value

    @inbounds while i <= N
        v1 = X[i, b];
        v1s = v1*v1
        y = v1s - c_p2;
        t = s_p2 + y;
        c_p2 = (t - s_p2) - y;
        s_p2 = t
        v1q = v1s^q
        y = v1q - c_mq;
        t = s_mq + y;
        c_mq = (t - s_mq) - y;
        s_mq = t

        j = i + nthr
        if j <= N
            v2 = X[j, b];
            v2s = v2*v2
            y = v2s - c_p2;
            t = s_p2 + y;
            c_p2 = (t - s_p2) - y;
            s_p2 = t
            v2q = v2s^q
            y = v2q - c_mq;
            t = s_mq + y;
            c_mq = (t - s_mq) - y;
            s_mq = t
        end
        i += g
    end

    smem[tid] = s_p2
    smem[tid+off_cp2] = c_p2
    smem[tid+off_mq] = s_mq
    smem[tid+off_cmq] = c_mq
    sync_threads()

    off = nthr >>> 1
    while off > 0
        if tid <= off
            smem[tid] += smem[tid+off]
            smem[tid+off_cp2] += smem[tid+off_cp2+off]
            smem[tid+off_mq] += smem[tid+off_mq+off]
            smem[tid+off_cmq] += smem[tid+off_cmq+off]
        end
        sync_threads()
        off >>>= 1
    end
    if tid == 1
        out_p2[blockIdx().x, b] = smem[1] + smem[1+off_cp2]
        out_mq[blockIdx().x, b] = smem[1+off_mq] + smem[1+off_cmq]
    end
    return
end

"Emit partial matrices (blocks_x×B) for p2 and (x^2)^q to be finished on the host."
# Convenience: default m4 (calls Val(2))
reduce_cols_2moments_partials!(
    X::CuArray{Float64,2};
    threads::Int = 256,
    stream = nothing,
) = reduce_cols_2moments_partials!(X, Val(2); threads = threads, stream = stream)

# Val(2) → reuse the optimized m4 kernel
function reduce_cols_2moments_partials!(
    X::CuArray{Float64,2},
    ::Val{2};
    threads::Int = 256,
    stream = nothing,
)
    N, B = size(X)
    blocks_x = min(1024, cld(N, threads*2));
    blocks_y = B
    out_p2 = CuArray{Float64}(undef, blocks_x, B)
    out_m4 = CuArray{Float64}(undef, blocks_x, B)
    shmem = sizeof(Float64) * threads * 4
    if stream === nothing
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem k_reduce_cols_2moments!(
            X,
            out_p2,
            out_m4,
        )
    else
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem stream=stream k_reduce_cols_2moments!(
            X,
            out_p2,
            out_m4,
        )
    end
    return out_p2, out_m4
end

# Integer Val(Q≠2) → compile-time multiply chain
function reduce_cols_2moments_partials!(
    X::CuArray{Float64,2},
    ::Val{Q};
    threads::Int = 256,
    stream = nothing,
) where {Q<:Number}
    @assert Q >= 0
    N, B = size(X)
    blocks_x = min(1024, cld(N, threads*2));
    blocks_y = B
    out_p2 = CuArray{Float64}(undef, blocks_x, B)
    out_mq = CuArray{Float64}(undef, blocks_x, B)
    shmem = sizeof(Float64) * threads * 4
    if stream === nothing
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem k_reduce_cols_2moments_qv!(
            X,
            out_p2,
            out_mq,
            Val(Q),
        )
    else
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem stream=stream k_reduce_cols_2moments_qv!(
            X,
            out_p2,
            out_mq,
            Val(Q),
        )
    end
    return out_p2, out_mq
end

"""
    fwht_batched_auto!(x; max_tile_L=10, threads=256, stream=nothing)

Batched FWHT on the columns of `x` (size N×B, N = 2^L), using a tiled
"head + tail" scheme:

  * We choose a tile size `tile = 2^tile_L`, where
        tile_L = min(max_tile_L, L)
    so the tile is always a power of two and divides N.

  * The first `fuse_stages = tile_L` stages are done in `fwht_head_batched!`
    inside each tile in shared memory (fast, lots of reuse).

  * Remaining stages (if any) are done in `fwht_tail_batched!` on the full
    vector in global memory.

Why `max_tile_L = 10` by default?

  * `tile = 2^10 = 1024` elements per tile.
  * For Float64, that’s 1024 * 8 bytes = 8 KiB of shared memory per block,
    which is small enough to fit comfortably on most GPUs.
  * 1024 is a good compromise: big enough to amortize loads, small enough
    to keep decent occupancy.

For small vectors (L ≤ max_tile_L), we set tile = N and fuse all stages
in the head, so no tail kernel is needed.
"""
function fwht_batched_auto!(
    x::CuArray{Float64,2};
    max_tile_L::Int = 10,
    threads::Int = 256,
    stream = nothing,
)
    N, B = size(x)
    @assert ispow2(N) "FWHT requires N to be a power of two"
    L = trailing_zeros(N)  # log2(N)

    # Choose tile size: tile = 2^tile_L, with tile_L ≤ L and ≤ max_tile_L.
    # This guarantees tile divides N, and (1<<tile_L) == tile.
    tile_L = min(max_tile_L, L)
    tile = 1 << tile_L
    fuse_stages = tile_L # fuse as many stages as fit in a tile

    if stream === nothing
        # Head: do the first `fuse_stages` FWHT stages per tile in shared memory.
        fwht_head_batched!(x; tile = tile, fuse_stages = fuse_stages, threads = threads)

        # Tail: if there are remaining stages (L > fuse_stages), do those globally.
        if fuse_stages < L
            fwht_tail_batched!(x; start_stage = fuse_stages, threads = threads)
        end
    else
        fwht_head_batched!(
            x;
            tile = tile,
            fuse_stages = fuse_stages,
            threads = threads,
            stream = stream,
        )
        if fuse_stages < L
            fwht_tail_batched!(
                x;
                start_stage = fuse_stages,
                threads = threads,
                stream = stream,
            )
        end
    end

    return x
end

"""
    fwht_batched_auto_reduce_partials!(x, q; max_tile_L=10, threads=256, stream=nothing)

Like `fwht_batched_auto!`, but instead of calling a separate reduction kernel
it fuses the *last* FWHT stage with a 2-moment reduction and returns the small
partial matrices `(out_p2, out_mq)` of size (blocks_x × B).

- `x` is modified in-place and ends up FWHT-transformed.
- `q` is a `Val{Q}`: use `Val(2.0)` for q=2, etc.
"""
function fwht_batched_auto_reduce_partials!(
    x::CuArray{Float64,2},
    q::Val{Q};
    max_tile_L::Int = 10,
    threads::Int = 256,
    stream = nothing,
) where {Q}
    N, B = size(x)
    @assert ispow2(N) "FWHT requires N to be a power of two"
    L = trailing_zeros(N)  # log2(N)

    # same tiling choice as fwht_batched_auto!
    tile_L = min(max_tile_L, L)
    tile = 1 << tile_L
    fuse_stages = tile_L # number of stages done in the head
    start_stage = fuse_stages      # first tail stage index

    # head: fused early stages in shared memory
    if stream === nothing
        fwht_head_batched!(x; tile = tile, fuse_stages = fuse_stages, threads = threads)
    else
        fwht_head_batched!(
            x;
            tile = tile,
            fuse_stages = fuse_stages,
            threads = threads,
            stream = stream,
        )
    end

    # tail + reduction
    if fuse_stages == L
        # no tail: all stages already done in shared memory
        # → just do the usual reduction kernel as a fallback
        return reduce_cols_2moments_partials!(x, q; threads = threads, stream = stream)
    else
        # some stages remain: do tail stages and fuse the last one with reduction
        return fwht_tail_and_reduce_partials!(
            x,
            q;
            start_stage = start_stage,
            threads = threads,
            stream = stream,
        )
    end
end

function fwht_batched_auto_reduce_partials!(
    x::CuArray{Float64,2},
    out_p2::CuArray{Float64,2},
    out_mq::CuArray{Float64,2},
    q::Val{Q};
    max_tile_L::Int = 10,
    threads::Int = 256,
    stream = nothing,
) where {Q}
    N, B = size(x)
    @assert ispow2(N) "FWHT requires N to be a power of two"
    L = trailing_zeros(N)
    @assert size(out_p2, 2) ≥ B
    @assert size(out_mq, 2) ≥ B

    # same tiling choice as fwht_batched_auto!
    tile_L = min(max_tile_L, L)
    tile = 1 << tile_L
    fuse_stages = tile_L
    start_stage = fuse_stages

    # head: fused early stages in shared memory
    if stream === nothing
        fwht_head_batched!(x; tile = tile, fuse_stages = fuse_stages, threads = threads)
    else
        fwht_head_batched!(
            x;
            tile = tile,
            fuse_stages = fuse_stages,
            threads = threads,
            stream = stream,
        )
    end

    # tail + reduction
    if fuse_stages == L
        # all stages done in shared: fall back to pure reduction
        return reduce_cols_2moments_partials2!(
            x,
            out_p2,
            out_mq,
            q;
            threads = threads,
            stream = stream,
        )
    else
        # do remaining stages and fuse last one with reduction
        return fwht_tail_and_reduce_partials2!(
            x,
            out_p2,
            out_mq,
            q;
            start_stage = start_stage,
            threads = threads,
            stream = stream,
        )
    end
end

# convenience for integer q (e.g. q = Val(2))
function fwht_batched_auto_reduce_partials!(
    x::CuArray{Float64,2},
    ::Val{Q};
    max_tile_L::Int = 10,
    threads::Int = 256,
    stream = nothing,
) where {Q<:Integer}
    return fwht_batched_auto_reduce_partials!(
        x,
        Val(Float64(Q));
        max_tile_L = max_tile_L,
        threads = threads,
        stream = stream,
    )
end

function reduce_cols_2moments_partials2!(
    X::CuArray{Float64,2},
    out_p2::CuArray{Float64,2},
    out_mq::CuArray{Float64,2},
    q::Val{Q};
    threads::Int = 256,
    stream = nothing,
) where {Q}
    N, B = size(X)
    @assert size(out_p2, 2) ≥ B "out_p2 has too few columns"
    @assert size(out_mq, 2) ≥ B "out_mq has too few columns"

    blocks_x = size(out_p2, 1)        # expect caller to size as min(1024, cld(N,threads*2))
    blocks_y = B
    shmem = sizeof(Float64) * threads * 4

    if stream === nothing
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem k_reduce_cols_2moments_qv!(
            X,
            out_p2,
            out_mq,
            q,
        )
    else
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem stream=stream k_reduce_cols_2moments_qv!(
            X,
            out_p2,
            out_mq,
            q,
        )
    end
    return out_p2, out_mq
end

function reduce_cols_2moments_partials2!(
    X::CuArray{Float64,2},
    out_p2::CuArray{Float64,2},
    out_mq::CuArray{Float64,2},
    ::Val{2};
    threads::Int = 256,
    stream = nothing,
)
    N, B = size(X)
    @assert size(out_p2, 2) ≥ B
    @assert size(out_mq, 2) ≥ B

    blocks_x = size(out_p2, 1)
    blocks_y = B
    shmem = sizeof(Float64) * threads * 4

    if stream === nothing
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem k_reduce_cols_2moments!(
            X,
            out_p2,
            out_mq,
        )   # out_mq is m4 in this case
    else
        @cuda threads=threads blocks=(blocks_x, blocks_y) shmem=shmem stream=stream k_reduce_cols_2moments!(
            X,
            out_p2,
            out_mq,
        )
    end
    return out_p2, out_mq
end

function compute_chunk_sre_cuda_batched(
    istart::Int,
    iend::Int,
    ψ,
    Zwhere::Vector{Int},
    XTAB::Vector{UInt64};
    q::Val{Q} = Val(2),
    batch::Int = 64,
    threads::Int = 256,
)::Tuple{Float64,Float64} where {Q}

    # before: N = length(data(ψ)); ψd = CuArray(data(ψ))
    ψh = collect(ComplexF64, data(ψ)) # materialize to a plain Vector{ComplexF64}
    N = length(ψh)
    ψd = CuArray(ψh)

    # work buffers
    X = CuArray{Float64}(undef, N, batch) # (dim, B)
    masks_h = CUDA.pin(Vector{UInt64}(undef, batch))
    masks_d = CuArray{UInt64}(undef, batch)

    p2SAM = 0.0
    mSAM = 0.0

    blocks_x = min(1024, cld(N, threads*2))
    out_p2 = CuArray{Float64}(undef, blocks_x, batch)
    out_m4 = CuArray{Float64}(undef, blocks_x, batch)

    # walk masks on the host, push B at a time
    cur_mask = UInt64(XTAB[istart])
    i = istart
    while i <= iend
        # prepare up to `batch` masks
        fillcount = 0
        while i <= iend && fillcount < batch
            if i == istart && fillcount == 0
                masks_h[1] = cur_mask
            else
                site = Zwhere[i-1] - 1
                cur_mask ⊻= (UInt64(1) << site)
                masks_h[fillcount+1] = cur_mask
            end
            i += 1
            fillcount += 1
        end

        # upload masks (no views!)
        CUDA.copyto!(masks_d, 1, masks_h, 1, fillcount)

        # BUILD → FWHT for the active columns only
        # (launch batched kernels with blocks=(blocks_x, fillcount))
        build_inVR_batched!(ψd, masks_d, view(X, :, 1:fillcount); threads = threads)
        # fwht_batched!(view(X, :, 1:fillcount); threads=threads)

        Xsub = view(X, :, 1:fillcount)

        # FWHT + last-stage reduction fused
        out_p2_batch, out_mq_batch = fwht_batched_auto_reduce_partials!(
            Xsub,
            view(out_p2, :, 1:fillcount),
            view(out_m4, :, 1:fillcount),
            q;
            max_tile_L = 10,
            threads = threads,
        )

        # host fold (same style as before)
        @views p2SAM += sum(sum(Array(out_p2_batch), dims = 1))[]
        @views mSAM += sum(sum(Array(out_mq_batch), dims = 1))[]
    end

    return (p2SAM, mSAM)
end

function compute_chunk_sre_cuda_batched!(
    ws::SREChunkWorkspace,
    istart::Int,
    iend::Int,
    Zwhere::Vector{Int},
    XTAB::Vector{UInt64};
    q::Val{Q} = Val(2),
    batch::Int = size(ws.X, 2),
    threads::Int = ws.threads,
)::Tuple{Float64,Float64} where {Q}

    ψd = ws.ψd
    X = ws.X
    masks_h = ws.masks_h
    masks_d = ws.masks_d
    out_p2 = ws.out_p2
    out_mq = ws.out_mq

    N = length(ψd)
    @assert size(X, 1) == N
    batch = min(batch, size(X, 2))

    p2SAM = 0.0
    mSAM = 0.0

    cur_mask = UInt64(XTAB[istart])
    i = istart

    while i <= iend
        # how many columns in this batch
        fillcount = 0
        while i <= iend && fillcount < batch
            idx = fillcount + 1
            if i == istart && fillcount == 0
                masks_h[1] = cur_mask
            else
                site = Zwhere[i-1] - 1
                cur_mask ⊻= (UInt64(1) << site)
                masks_h[idx] = cur_mask
            end
            i += 1
            fillcount += 1
        end

        # upload masks for this batch
        CUDA.copyto!(masks_d, 1, masks_h, 1, fillcount)

        # views for the active columns
        Xsub = @view X[:, 1:fillcount]
        out_p2_sub = @view out_p2[:, 1:fillcount]
        out_mq_sub = @view out_mq[:, 1:fillcount]

        # BUILD on GPU
        build_inVR_batched!(ψd, masks_d, Xsub; threads = threads)

        # FWHT (head + tail) + fused last-stage reduction → out_p2_sub, out_mq_sub
        fwht_batched_auto_reduce_partials!(
            Xsub,
            out_p2_sub,
            out_mq_sub,
            q;
            max_tile_L = 10,
            threads = threads,
        )

        # --- GPU fold: sum all partials per batch on device ---
        # This launches a small reduction kernel on the GPU and returns a scalar.
        p2SAM += CUDA.sum(out_p2_sub)
        mSAM += CUDA.sum(out_mq_sub)
    end

    return (p2SAM, mSAM)
end

function SRE(ψ, q::Val{Q}; progress = false, batch::Int = 128, threads::Int = 256) where {Q}
    if progress
        @warn "Progress bar not implemented for CUDA backend."
    end

    L = qubits(ψ)
    dim = 1 << L

    XTAB, Zwhere = generate_gray_table(L, 2)

    # prepare workspace
    ws = SREChunkWorkspace(ψ; max_batch = batch, threads = threads)

    # GPU path for exponent Val{Q}
    p2SAM, m2SAM = compute_chunk_sre_cuda_batched!(
        ws,
        1,
        length(XTAB),
        Zwhere,
        XTAB;
        q = q,
        batch = batch,
        threads = threads,
    )

    return (-log2(m2SAM/dim), abs(1 - p2SAM/dim))
end

SRE(ψ, q::Integer; kwargs...) = SRE(ψ, Val(q); kwargs...)
SRE(ψ, q::AbstractFloat; kwargs...) = SRE(ψ, Val(q); kwargs...)

end
