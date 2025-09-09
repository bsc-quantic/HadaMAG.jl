module HadaMAGMPIExt

using HadaMAG
using MPI

# This module provides the low-level kernels for the MPI backend.
"""
    generate_binary(n::Int, comm::MPI.Comm = MPI.COMM_WORLD)
        → (local_codes, local_flips, offset)

Generate the length-`n` binary Gray sequence **slice** for this rank in `comm`.

Internally, the global sequence of size `2^n` is partitioned across all ranks of `comm` (via `HadaMAG.partition_counts`), and each rank locally computes:

 1. `local_codes[j] = k ⊻ (k >> 1)` for its assigned global indices `k = offset, offset+1, …`.
 2. `local_flips[j]` as the 1‑based bit position that flipped between successive codes—
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
    threaded_chunk_reduce(len, compute_chunk;
                               progress=false, progress_stride=0, hz=8, io=stderr)

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
)
    nthr = Threads.nthreads()
    tcnt, tdisp = HadaMAG.partition_counts(len, nthr)
    acc = fill((0.0, 0.0), nthr)

    p =
        progress ? HadaMAG.CounterProgress(len; hz = hz, io = io, tty = true) :
        HadaMAG.NoProgress()

    # tiny adaptor so existing 3-arg compute_chunk keeps working
    @inline function call_chunk(f, tid, istart, iend)
        if progress && progress_stride > 0
            return f(tid, istart, iend, p, progress_stride)
        else
            return f(tid, istart, iend, p, 0) # 0 means no progress
        end
    end

    Threads.@threads :static for _ = 1:nthr
        tid = Threads.threadid()
        istart = tdisp[tid] + 1
        iend = tdisp[tid] + tcnt[tid]
        if tcnt[tid] == 0
            acc[tid] = (0.0, 0.0)
        else
            acc[tid] = call_chunk(compute_chunk, tid, istart, iend)
            # coarse progress: count this thread's whole slice as done
            if progress && progress_stride == 0
                HadaMAG.tick!(p, iend - istart + 1)
            end
        end
    end

    HadaMAG.finish!(p)

    a = 0.0;
    b = 0.0
    @inbounds for (a1, b1) in acc
        a += a1;
        b += b1
    end
    return a, b
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

    # Allocate the *full* scratch arrays…
    TMP1 = Vector{Float64}(undef, dim)
    TMP2 = Vector{Float64}(undef, dim)
    Xvec1 = Vector{Float64}(undef, dim)
    Xvec2 = Vector{Float64}(undef, dim)

    # … and initialize them in parallel
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

    # Local threaded work — no copies of TMP1/TMP2
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
    # node_vals = MPI.Allreduce([PS2, MS2], MPI.SUM, shm_comm)
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
    return (-log2(m2SAM / N), 1.0 - p2SAM / N)
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
    progress_stride = progress ? max(div(length(Nsamples), 100), 10) : 0 # update ~100 times

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

function mana_SRE2(ψ)
    dim = length(data(ψ))
    L = qudits(ψ)
    Ncores = Threads.nthreads() # Number of threads

    XTAB, Zwhere = generate_gray_table(L, 3)

    # here we know where to act with X_j operator to go from Pauli string number ix, to PS numbered ix+1
    # also we store XTAB which tells us what Pauli string is at position ix
    # constistencY; XTAB[ix] ^= XTAB[ix+1] -- only single 1 at position Zwhere[ix]
    p2SAM = m2SAM = m3SAM = 0.0

    # divide the gray's code (which has 2^N positions) into Ncores, more or less equal patches
    # we store the starting and finishing index corresponding to each of the patches in istart and iend
    istart = [div((i-1)*(length(Zwhere)+1), Ncores) + 1 for i = 1:Ncores]
    iend = [div(i*(length(Zwhere)+1), Ncores) for i = 1:Ncores]

    # The last thread processes until the end of Z (here we mimic Z.size()+1 from C++ by adding one extra element, if needed)
    PS2 = zeros(Float64, Ncores)
    MS2 = zeros(Float64, Ncores)
    MS3 = zeros(ComplexF64, Ncores)

    for j = 1:Ncores
        # Each thread/processes the subrange [istart[j], iend[j]-1]
        ps2, ms2, ms3 = HadaMAG._compute_chunk_mana_SRE(istart[j], iend[j], ψ, Zwhere, XTAB)
        PS2[j] = ps2
        MS2[j] = ms2
        MS3[j] = ms3
    end

    # Now sum the partial sums to obtain the final results.
    p2SAM = sum(PS2)
    m2SAM = sum(MS2)
    m3SAM = sum(MS3)

    println("mana = ", log2(p2SAM)/log2(3.0))

    # SRE is returned here    return (-log2(m2SAM/dim)/log2(3.0), -1.0*log2(real(m3SAM)/dim)/log2(3.0))
end

end
