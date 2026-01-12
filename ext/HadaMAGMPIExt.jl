module HadaMAGMPIExt # This module provides the low-level kernels for the MPI backend.

using HadaMAG
using MPI

"""
    generate_binary(n::Int, comm::MPI.Comm = MPI.COMM_WORLD)
        → (local_codes, local_flips, offset)

Generate the length-`n` binary Gray sequence **slice** for this rank in `comm`.

Internally, the global sequence of size `2^n` is partitioned across all ranks of `comm` (via `HadaMAG.partition_counts`), and each rank locally computes:

 1. `local_codes[j] = k ⊻ (k >> 1)` for its assigned global indices `k = offset, offset+1, ...`.
 2. `local_flips[j]` as the 1‑based bit position that flipped between successive codes,
    including the flip from the preceding rank’s last code into this rank’s first.

This requires only an (O(P)) broadcast of counts/displacements and (O(2^n/P)) local work, with no large scatter.

# Arguments
- `n::Int`
  Number of bits in the Gray sequence (global length is (2^n)).
- `comm::MPI.Comm`
  MPI communicator over which to split the sequence (defaults to `MPI.COMM_WORLD`).

# Returns
- `local_codes::Vector{UInt64}`
  The Gray codes for indices `offset : offset+count-1`.
- `local_flips::Vector{Int}`
  Same length as `local_codes`;
  `local_flips[j]` is the 1‑based index of the bit that flipped going
  from `local_codes[j-1] → local_codes[j]` (for `j==1`, from the global predecessor).
- `offset::Int`
  The 0‑based starting index of this slice in the full Gray sequence.
"""
function generate_binary(n::Int, comm = MPI.COMM_WORLD)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)
    return HadaMAG.generate_binary_splitted(n, rank, P)
end

"""
    generate_general(n::Int, d::Int; comm = MPI.COMM_WORLD)
        → (local_codes, local_flips, offset)

Generate the length-`n` *d*-ary Gray sequence **slice** for this rank in `comm`.
"""
function generate_general(n::Int, d::Int, comm = MPI.COMM_WORLD)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)
    return HadaMAG.generate_general_splitted(n, d, rank, P)
end

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

    p =
        progress ? HadaMAG.CounterProgress(len; hz = hz, io = io, tty = true) :
        HadaMAG.NoProgress()

    # tiny adaptor (keeps your 5-arg signature)
    @inline function call_chunk(f, tid, istart, iend)
        if progress && progress_stride > 0
            return f(tid, istart, iend, p, progress_stride)
        else
            return f(tid, istart, iend, p, 0) # 0 means no progress
        end
    end

    Threads.@threads :static for tid = 1:nthr
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

threaded_chunk_reduce(f::Function, len::Integer; kwargs...) =
    threaded_chunk_reduce(len, f; kwargs...)
threaded_chunk_reduce(len::Integer, f::Function, ::MPI.Comm; kwargs...) =
    threaded_chunk_reduce(len, f; kwargs...)
threaded_chunk_reduce(f::Function, len::Integer, ::MPI.Comm; kwargs...) =
    threaded_chunk_reduce(len, f; kwargs...)

# Scatter a contiguous range 1:n over MPI ranks,
# returning your local index-range and the displs if you need them
function scatter_range(n::Int, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)
    counts, displs = HadaMAG.partition_counts(n, P)
    istart = displs[rank+1] + 1
    iend = displs[rank+1] + counts[rank+1]
    return istart:iend, (counts, displs)
end

@fastmath function SRE(ψ, q; progress::Bool = true)
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank, nprocs = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    L = qubits(ψ)
    dim = 1 << L

    XTAB, Zwhere, displs, _ = generate_binary(L, comm)

    # Split out a node‐local (shared‐memory) communicator
    shm_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    shm_rank = MPI.Comm_rank(shm_comm)

    # Allocate all the scratch arrays
    TMP1 = Vector{Float64}(undef, dim)
    TMP2 = Vector{Float64}(undef, dim)
    Xvec1 = Vector{Float64}(undef, dim)
    Xvec2 = Vector{Float64}(undef, dim)

    # initialize them in parallel
    Threads.@threads for i = 1:dim
        r = real(ψ[i])
        im = imag(ψ[i])
        Xvec1[i] = r
        Xvec2[i] = im
        TMP1[i] = r + im
        TMP2[i] = im - r
    end

    # One `inVR` per thread (thread-local scratch)
    scratch_inVR = [Vector{Float64}(undef, dim) for _ = 1:Threads.nthreads()]

    progress_stride = progress ? max(div(length(XTAB), 100), 10) : 0 # update ~100 times

    # Local threaded work (no copies of TMP1/TMP2)
    PS2, MS2 = threaded_chunk_reduce(
        length(XTAB);
        progress = (progress && rank == 0),
        progress_stride,
    ) do tid, istart, iend, p_, stride
        HadaMAG.compute_chunk_sre(
            istart,
            iend,
            ψ,
            q,
            Zwhere,
            XTAB,
            TMP1,
            TMP2,
            Xvec1,
            Xvec2,
            scratch_inVR[tid],
            p_,
            stride,
        )
    end

    # Pack + intra-node reduction
    buf = [PS2, MS2]
    MPI.Allreduce!(buf, MPI.SUM, shm_comm)   # in-place
    node_vals = buf

    # Inter-node reduction by leaders
    color = (shm_rank == 0 ? 0 : nothing)
    inter_comm = MPI.Comm_split(comm, color, rank)
    global_leader = Vector{Float64}(undef, 2)
    if shm_rank == 0
        global_leader .= MPI.Allreduce(node_vals, MPI.SUM, inter_comm)
    end

    # Broadcast to node
    MPI.Bcast!(global_leader, 0, shm_comm)
    p2SAM, m2SAM = Tuple(global_leader)

    N = length(data(ψ))
    return (-log2(m2SAM / N), abs(1.0 - p2SAM / N))
end

function MC_SRE(
    ψ,
    q,
    Nβ::Int,
    Nsamples::Int;
    seed::Union{Nothing,Int} = nothing,
    progress = true,
)
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    base_seed = (rank == 0 ? (seed === nothing ? rand(1:(10^9)) : seed) : 0)
    base_seed = MPI.Bcast(base_seed, 0, comm)

    X = data(ψ)
    dim = length(X)
    L = qubits(ψ)

    # shared tmp1/tmp2 per rank
    tmp1 = Vector{Float64}(undef, dim)
    tmp2 = Vector{Float64}(undef, dim)
    HadaMAG.build_tmp!(tmp1, tmp2, X)

    # per-thread scratch
    nthr = Threads.nthreads()
    scratch = [Vector{Float64}(undef, dim) for _ = 1:nthr]

    p =
        (progress && rank == 0) ?
        HadaMAG.CounterProgress(Nsamples; hz = 8, io = stderr, tty = true) :
        HadaMAG.NoProgress()
    progress_stride = progress ? max(div(Nsamples, 100), 10) : 0 # update ~100 times

    # local β block
    idx_range, _ = scatter_range(Nβ, comm)
    β_vals = Float64.((idx_range .- 1) ./ (Nβ - 1))

    local_M2 = Vector{Float64}(undef, length(β_vals))
    local_M2ADD = Vector{Float64}(undef, length(β_vals))
    local_P2 = Vector{Float64}(undef, length(β_vals))

    Threads.@threads :static for k in eachindex(β_vals)
        tid = Threads.threadid()
        β = β_vals[k]
        j = first(idx_range) + k - 1
        m2_mean, _m2sq, m2add_mean, p2_mean, _n = HadaMAG.mc_sre_β!(
            X,
            tmp1,
            tmp2,
            scratch[tid],
            q,
            Nsamples,
            base_seed + j,
            β,
            L,
            p,
            progress_stride,
        )
        local_M2[k] = m2_mean
        local_M2ADD[k] = m2add_mean
        local_P2[k] = p2_mean
    end

    # helper: Gatherv to root using VBuffer
    gather_vec(v) = begin
        nloc = length(v)
        counts = MPI.gather(nloc, comm; root = 0)
        if rank == 0
            counts32 = Int32.(counts)
            displs32 = Int32[0; cumsum(counts32[1:(end-1)])]
            total = sum(counts32)
            out = Vector{Float64}(undef, total)
            MPI.Gatherv!(v, MPI.VBuffer(out, counts32, displs32), comm; root = 0)
            out
        else
            MPI.Gatherv!(v, nothing, comm; root = 0)
            Float64[]
        end
    end

    all_M2 = gather_vec(local_M2)
    all_M2ADD = gather_vec(local_M2ADD)
    all_P2 = gather_vec(local_P2)

    m2 = 0.0
    if rank == 0
        Nβ % 2 == 1 || error("Simpson needs an odd number of β points.")
        x = collect((0:(Nβ-1)) ./ (Nβ-1))
        I_res = HadaMAG.integrate_simpson_uniform(x, all_M2)
        I_m2ADD = HadaMAG.integrate_simpson_uniform(x, all_M2ADD)
        m2 = -log2(2.0^(-I_res) + I_m2ADD)
    end
    m2 = MPI.Bcast(m2, 0, comm)

    return m2
end

@fastmath function Mana(ψ::StateVec{ComplexF64,3}; progress::Bool = true)
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank, nprocs = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    L = qudits(ψ)
    dim = 3^L

    XTAB, Zwhere, displs, _ = generate_general(L, 3, comm)

    # Split out a node‐local (shared‐memory) communicator
    shm_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    shm_rank = MPI.Comm_rank(shm_comm)

    AA = Vector{Int}(undef, dim)
    Threads.@threads for i = 1:dim
        x = i - 1 # 0-based index
        y = 0
        pow = 1
        @inbounds for _ = 1:L
            d = x % 3
            x ÷= 3
            nd = (3 - d) % 3 # 0 -> 0, 1 -> 2, 2 -> 1
            y += nd * pow
            pow *= 3
        end
        @inbounds AA[i] = y + 1 # back to 1-based
    end

    # One `inV` per thread (thread-local scratch)
    scratch_inV = [Vector{ComplexF64}(undef, dim) for _ = 1:Threads.nthreads()]

    progress_stride = progress ? max(div(size(XTAB, 2), 100), 10) : 0 # update ~100 times

    # Local threaded work (no copies of AA)
    p2SAM = threaded_chunk_reduce(
        size(XTAB, 2);
        nout = 1,
        progress = (progress && rank == 0),
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

    buf = [only(p2SAM)]

    # -------- Intra-node reduction (shared-memory communicator) --------
    MPI.Allreduce!(buf, MPI.SUM, shm_comm)
    node_vals = buf

    # -------- Inter-node reduction among node leaders --------
    color = (shm_rank == 0 ? 0 : nothing)
    inter_comm = MPI.Comm_split(comm, color, rank)
    global_leader = Vector{Float64}(undef, 1)
    if shm_rank == 0
        global_leader .= MPI.Allreduce(node_vals, MPI.SUM, inter_comm)
    end

    # Broadcast to all node members
    MPI.Bcast!(global_leader, 0, shm_comm)
    p2SAM = only(global_leader)

    return log2(p2SAM)/log2(3.0)
end

end
