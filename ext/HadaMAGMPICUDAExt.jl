module HadaMAGMPICUDAExt # This module provides the low-level kernels for the MPI backend.

using HadaMAG
using MPI
using CUDA

# helper: split 1:N into k nearly equal contiguous ranges
function split_ranges(N::Int, k::Int)
    k = max(1, min(k, N))
    q, r = divrem(N, k)
    ranges = Vector{UnitRange{Int}}(undef, k)
    start = 1
    for i in 1:k
        len = q + (i <= r ? 1 : 0)
        stop = start + len - 1
        ranges[i] = start:stop
        start = stop + 1
    end
    ranges
end

# Split this rank's chunk across its GPUs
function split_range_into_subranges(r::UnitRange{Int}, parts::Int)
    # split_ranges already balances 1:length(r)
    loc = split_ranges(length(r), parts)
    base = first(r) - 1
    return [(base + first(rr)):(base + last(rr)) for rr in loc ]
end

function SRE(ψ, q::Val{Q}; progress = false, batch::Int = 128, threads::Int = 256) where {Q}
    MPI.Initialized() || MPI.Init() # Make sure MPI is live
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0 && progress
        @warn "Progress bar not implemented for MPI+CUDA backend."
    end

    HadaMAGCUDAExt = Base.get_extension(HadaMAG, :HadaMAGCUDAExt)

    # Global problem setup (replicated on every rank)
    L = qubits(ψ)
    dim = 1 << L
    XTAB, Zwhere = generate_gray_table(L, 2)
    nMasks = length(XTAB)

    # Split masks across MPI ranks (nodes)
    global_ranges = split_ranges(nMasks, nprocs)
    r_global = global_ranges[rank + 1]   # Julia is 1-based

    # GPUs visible on THIS rank/node
    devs = collect(CUDA.devices())
    ngpu = length(devs)

    subranges = split_range_into_subranges(r_global, ngpu)
    local_ngpu = length(subranges)

    # One workspace per GPU
    workspaces = Vector{HadaMAGCUDAExt.SREChunkWorkspace}(undef, ngpu)
    for (i, dev) in enumerate(devs)
        CUDA.device!(dev)
        workspaces[i] = HadaMAGCUDAExt.SREChunkWorkspace(ψ; max_batch=batch, threads=threads)
    end

    partials = Vector{Tuple{Float64,Float64}}(undef, local_ngpu)

    Threads.@sync for i in 1:local_ngpu
        dev = devs[i]
        r_sub = subranges[i]
        ws = workspaces[i]

        Threads.@spawn begin
            CUDA.device!(dev)

            p2_i, mq_i = HadaMAGCUDAExt.compute_chunk_sre_cuda_batched!(
                ws,
                first(r_sub), last(r_sub),
                Zwhere, XTAB;
                q = q,
                batch = batch,
                threads = threads,
            )

            # synchronize()  # explicit device sync
            partials[i] = (p2_i, mq_i)
        end
    end

    # Sum within this node/rank
    p2_local = sum(t -> t[1], partials)
    mq_local = sum(t -> t[2], partials)

    # Global reduction across all ranks
    p2_total = MPI.Allreduce(p2_local, +, comm)
    mq_total = MPI.Allreduce(mq_local, +, comm)

    # Final observables
    SRE = -log2(mq_total / dim)
    L2_dev = abs(1 - p2_total / dim)

    return (SRE, L2_dev)
end

SRE(ψ, q::Integer; kwargs...) = SRE(ψ, Val(q); kwargs...)
SRE(ψ, q::AbstractFloat; kwargs...) = SRE(ψ, Val(q); kwargs...)

end
