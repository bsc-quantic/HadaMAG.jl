module ThreadedBackend # This module provides the low-level kernels for the Threaded backend

using LinearAlgebra
using Random
using HadaMAG

"""
    threaded_chunk_reduce(len, compute_chunk;
                               progress=false, progress_stride=0, hz=8, io=stderr, nout=2)

Static-scheduled threaded map-reduce over `len` items.

The iteration space is partitioned once and each Julia thread `tid`
receives a disjoint, contiguous subrange `[istart, iend]`. Your
`compute_chunk` is called exactly once per thread with those bounds so
you can index *thread-local* scratch by `tid`.

# Arguments
- `len::Integer`: number of logical items to process.
- `compute_chunk::Function`: called as compute_chunk(tid::Int, istart::Int, iend::Int, p, stride) (progress-aware).
 It must return a 2-tuple `(a, b)`; all per-thread tuples are reduced by **elementwise sum**.

# Keywords
- `progress::Bool=false`: enable a low-overhead progress indicator.
- `progress_stride::Int=0`: if `> 0`, the 5-arg form is used and you may
  call `HadaMAG.tick!(p, k)` inside `compute_chunk` every `k=stride`
  iterations; if `0`, a single coarse tick equal to the thread’s slice
  length is emitted after `compute_chunk` returns.
- `hz::Real=8`: max redraw rate for the progress indicator (ignored if
  `progress=false`).
- `io::IO=stderr`: stream for progress output.
- `nout::Int=2`: number of output values from `compute_chunk` (must be ≥ 1).

# Returns
The tuple `(a::Float64, b::Float64)`, elementwise sum of the
`compute_chunk` results from all threads.
"""
function threaded_chunk_reduce(
    len::Integer,
    compute_chunk::Function;
    progress::Bool = false,
    progress_stride::Int = 0,
    hz::Real = 8,
    io::IO = stderr,
    nout::Int = 2,
)
    @assert nout ≥ 1 "nout must be ≥ 1"

    nthr = Threads.nthreads()
    tcnt, tdisp = HadaMAG.partition_counts(len, nthr)

    # Accumulate per-thread results in a dense matrix (nout × nthr)
    acc = zeros(Float64, nout, nthr)

    p = progress ? HadaMAG.CounterProgress(len; hz = hz, io = io, tty = true) :
                   HadaMAG.NoProgress()

    # tiny adaptor (keeps your 5-arg signature)
    @inline function call_chunk(f, tid, istart, iend)
        if progress && progress_stride > 0
            return f(tid, istart, iend, p, progress_stride)
        else
            return f(tid, istart, iend, p, 0) # 0 means no progress
        end
    end

    Threads.@threads :static for tid in 1:nthr
        istart = tdisp[tid] + 1
        iend = tdisp[tid] + tcnt[tid]

        if tcnt[tid] == 0
            # leave acc[:, tid] as zeros
        else
            res = call_chunk(compute_chunk, tid, istart, iend)

            # Accept tuples or vectors; enforce length nout
            lenres = Base.length(res)
            @assert lenres == nout "compute_chunk must return $nout values; got $lenres"
            @inbounds for i = 1:nout
                acc[i, tid] = Float64(res[i])
            end

            # coarse tick if the chunk didn't use per-iteration progress
            if progress && progress_stride == 0
                HadaMAG.tick!(p, iend - istart + 1)
            end
        end
    end

    HadaMAG.finish!(p)

    # Reduce columns by sum, return as NTuple{nout,Float64}
    out = Vector{Float64}(undef, nout)
    @inbounds for i = 1:nout
        s = 0.0
        @simd for t = 1:nthr
            s += acc[i, t]
        end
        out[i] = s
    end
    return Tuple(out)
end

# Keep your convenience forwarders
threaded_chunk_reduce(f::Function, len::Integer; kwargs...) =
    threaded_chunk_reduce(len, f; kwargs...)

function SRE(ψ, q; progress::Bool = true)
    L = qubits(ψ)
    dim = 1 << L

    XTAB, Zwhere = generate_gray_table(L, 2)

    # Allocate all the necessary arrays
    TMP1 = zeros(Float64, dim)
    TMP2 = zeros(Float64, dim)
    Xloc1 = zeros(Float64, dim)
    Xloc2 = zeros(Float64, dim)

    # Fill all the arrays by multi-threading
    Threads.@threads for i = 1:dim
        r = real(ψ[i])
        im = imag(ψ[i])
        Xloc1[i] = r
        Xloc2[i] = im
        TMP1[i] = r + im
        TMP2[i] = im - r
    end

    # One `inVR` per thread (thread-local scratch)
    scratch_inVR = [zeros(Float64, dim) for _ = 1:Threads.nthreads()]

    progress_stride = progress ? max(div(length(XTAB), 100), 10) : 0 # update ~100 times

    # Local threaded work — no copies of TMP1/TMP2
    p2SAM, m2SAM = threaded_chunk_reduce(
        length(XTAB);
        progress,
        progress_stride,
    ) do tid, istart, iend, pbar, stride
        HadaMAG.compute_chunk_sre(
            istart,
            iend,
            ψ,
            q,
            Zwhere,
            XTAB,
            TMP1,
            TMP2,
            Xloc1,
            Xloc2,
            scratch_inVR[tid],
            pbar,
            stride,
        )
    end

    return (-log2(m2SAM/dim), abs(1-p2SAM/dim)) # TODO: should we really return 0.0 there?
end

function MC_SRE(
    ψ,
    q,
    Nβ::Int,
    Nsamples::Int;
    seed::Union{Nothing,Int} = nothing,
    progress = true,
)
    Nβ % 2 == 1 || error("Simpson needs an odd number of β points.")
    X = data(ψ)
    dim = length(X)
    L = qubits(ψ)

    # Precompute tmp1/tmp2 ONCE (shared, read-only)
    tmp1 = Vector{Float64}(undef, dim)
    tmp2 = Vector{Float64}(undef, dim)
    HadaMAG.build_tmp!(tmp1, tmp2, X)

    # Per-thread scratch for inVR
    nthr = Threads.nthreads()
    scratch_inVR = [zeros(Float64, dim) for _ = 1:nthr]

    # Outputs per β
    M2 = Vector{Float64}(undef, Nβ)
    M2ADD = Vector{Float64}(undef, Nβ)
    P2 = Vector{Float64}(undef, Nβ)

    base_seed = seed === nothing ? rand(1:(10^9)) : seed

    pbar =
        progress ? HadaMAG.CounterProgress(Nsamples; hz = 8, io = stderr) :
        HadaMAG.NoProgress()
    progress_stride = progress ? max(div(Nsamples, 100), 10) : 0 # update ~100 times

    Threads.@threads :static for i = 1:Nβ
        tid = Threads.threadid()
        β = (i - 1) / (Nβ - 1)
        m2_mean, _m2sq, m2add_mean, p2_mean, _n = HadaMAG.mc_sre_β!(
            X,
            tmp1,
            tmp2,
            scratch_inVR[tid],
            q,
            Nsamples,
            base_seed + i,
            β,
            L,
            pbar,
            progress_stride,
        )
        M2[i] = m2_mean
        M2ADD[i] = m2add_mean
        P2[i] = p2_mean
    end

    x = collect((0:(Nβ-1)) ./ (Nβ-1))
    I_res = HadaMAG.integrate_simpson_uniform(x, M2)
    I_m2ADD = HadaMAG.integrate_simpson_uniform(x, M2ADD)
    m2 = -log2(2.0^(-I_res) + I_m2ADD)
    return m2
end

function Mana(ψ; progress::Bool = true)
    L = qudits(ψ)
    dim = 3^L

    XTAB, Zwhere = HadaMAG.generate_general(L, 3)

    AA = Vector{Int}(undef, dim)
    Threads.@threads for i in 1:dim
        x = i - 1          # 0-based index
        y = 0
        pow = 1
        @inbounds for _ in 1:L
            d = x % 3
            x ÷= 3
            nd = (3 - d) % 3 # 0 -> 0, 1 -> 2, 2 -> 1
            y += nd * pow
            pow *= 3
        end
        @inbounds AA[i] = y + 1 # back to 1-based
    end

    # One `inV` per thread (thread-local scratch)
    scratch_inV = [zeros(ComplexF64, dim) for _ = 1:Threads.nthreads()]

    progress_stride = progress ? max(div(size(XTAB, 2), 100), 10) : 0 # update ~100 times

    # Local threaded work — no copies of TMP1/TMP2
    p2SAM = threaded_chunk_reduce(
        size(XTAB, 2);
        nout=1,
        progress,
        progress_stride,
    ) do tid, istart, iend, pbar, stride
        HadaMAG.compute_chunk_mana_qutrits(
            istart,
            iend,
            ψ,
            Zwhere,
            XTAB,
            AA,
            scratch_inV[tid],
            pbar,
            stride,
        )
    end
    p2SAM = only(p2SAM) # unpack single output

    return log2(p2SAM)/log2(3.0)
end

end # module ThreadedBackend
