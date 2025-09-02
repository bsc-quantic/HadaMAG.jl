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

function scatterv(A::AbstractVector{T}, comm::MPI.Comm; root::Int = 0) where {T}
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    # ─── 1) Root builds the counts & displs, then UBuffer/VBuffer ───────────
    if rank == root
        counts, displs = HadaMAG.partition_counts(length(A), P)

        # make a 2×P matrix: row1=counts, row2=displs
        sizes = vcat(vcat(counts...)', vcat(displs...)')       # 2×P Array{Int,2}
        size_ubuf = UBuffer(sizes, 2)            # each column is a NTuple{2,Int}
        vbuf = VBuffer(A, counts)
    else
        size_ubuf = UBuffer(nothing)
        vbuf = VBuffer(nothing)
    end

    # ─── 2) Scatter out each column of `sizes` as an NTuple{2,Int} ─────────
    local_count, local_disp =           # destructure the two‑element tuple
        MPI.Scatter(size_ubuf, NTuple{2,Int}, root, comm)

    # ─── 3) Now everyone allocates exactly the right 1‑D buffer ─────────────
    local_buf = zeros(T, local_count)

    # ─── 4) And scatter the data into it ───────────────────────────────────
    local_chunk = MPI.Scatterv!(vbuf, local_buf, root, comm)

    return local_chunk, local_disp
end

# Helper threaded map-reduce over chunks of a 1-D array
function threaded_chunk_reduce(len::Integer, f::Function, comm::MPI.Comm)
    nthr = Threads.nthreads()
    tcnt, tdisp = HadaMAG.partition_counts(len, nthr)

    # one slot per thread
    acc = Vector{Tuple{Float64,Float64}}(undef, nthr)
    @sync for tid = 1:nthr
        Threads.@spawn begin
            istart = tdisp[tid] + 1
            iend = tdisp[tid] + tcnt[tid]
            acc[tid] = f(istart, iend)
            # @show "Thread $tid processing range [$istart, $iend], result: $(acc[tid])"
        end
    end

    # now sum all the (a,b,c) tuples
    a, b = 0.0, 0.0, 0.0
    for (a₁, b₁) in acc
        a += a₁;
        b += b₁
    end
    return a, b
end

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

function MC_SRE2(
    ψ,
    Nβ::Int,
    Nsamples::Int,
    seed::Union{Nothing,Int};
    cleanup = true,
)
    # TODO: create a macro for this mpi rank assignment
    MPI.Initialized() || MPI.Init()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    comm = MPI.COMM_WORLD
    mpisize = MPI.Comm_size(comm)

    # Set a random seed
    seed = seed === nothing ? floor(Int, rand() * 1e9) : seed

    # Create a temporary directory for storing results
    tmpdir = rank == 0 ? mktempdir() : nothing
    tmpdir = MPI.bcast(tmpdir, 0, comm)

    beta_val(i) = Float64(i - 1) / (Nβ - 1) # so that β ∈ [0, 1]
    counts, displs = HadaMAG.partition_counts(Nβ, mpisize)
    r = rank + 1 # 1-based for arrays
    start = displs[r] + 1
    stop = displs[r] + counts[r]
    beta_rank = beta_val.(start:stop) # works even when counts[r]==0 (empty range)

    m2 = 0.0
    try
        # Each rank processes its own beta values
        for i = 1:length(beta_rank)
            β = beta_rank[i]
            # for i = 1:Nβ
            # β = Float64(i) / Nβ
            # convert beta value to j
            idx = (β * (Nβ - 1) |> round |> Int) + 1

            HadaMAG._compute_MC_SRE2_β(ψ, Nsamples, seed + idx, β, idx, tmpdir)
        end

        MPI.Barrier(MPI.COMM_WORLD)

        # Rank 0 computes the average of the results for each β
        if rank == 0
            x, res_means, res_stds, m2ADD_means, m2ADD_stds, naccepted =
                HadaMAG.process_files(seed; folder = tmpdir, Nβ)

            # Compute the final result using Simpson's rule
            integral_res = HadaMAG.integrate_simpson(x, res_means)
            integral_m2ADD = HadaMAG.integrate_simpson(x, m2ADD_means)

            m2 = -log2(2^(-integral_res) + integral_m2ADD)
        end
    finally
        if rank == 0
            # Rank 0 cleans up the temporary directory
            if cleanup
                rm(tmpdir, force = true, recursive = true)
            else
                @info "Retaining temp directory at: $tmpdir"
            end
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)

    # Broadcast the result to all ranks
    m2 = MPI.Bcast(m2, 0, comm)

    return m2
end

@fastmath function SRE(ψ, q)
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank, nprocs = MPI.Comm_rank(comm), MPI.Comm_size(comm)
    L = qubits(ψ)
    dim = 2^L

    local_xtab, local_zwhere, displs, _ = generate_binary(L, comm)

    # Split out a node‐local (shared‐memory) communicator
    shm_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    shm_rank = MPI.Comm_rank(shm_comm)

    # Allocate the *full* scratch arrays…
    TMP1 = zeros(Float64, dim)
    TMP2 = zeros(Float64, dim)
    Xloc1 = zeros(Float64, dim)
    Xloc2 = zeros(Float64, dim)

    # … and initialize them in parallel
    Threads.@threads for i = 1:dim
        r = real(ψ[i])
        im = imag(ψ[i])
        Xloc1[i] = r
        Xloc2[i] = im
        TMP1[i] = r + im
        TMP2[i] = im - r
    end

    # Do the local work
    PS2, MS2 = threaded_chunk_reduce(
        length(local_xtab),
        (i, j) -> HadaMAG._compute_chunk_SRE_v23(
            displs,
            i,
            j,
            ψ,
            q,
            local_zwhere,
            local_xtab,
            copy(TMP1),
            copy(TMP2),
            Xloc1,
            Xloc2,
        ),
        comm,
    )

    # Pack into one small vector
    local_vals = [PS2, MS2]

    # Do the cheap intra‑node reduction
    # All threads/processes on the same physical node
    node_vals = MPI.Allreduce(local_vals, MPI.SUM, shm_comm)

    # Split out inter‑node communicator
    color = (shm_rank == 0 ? 0 : nothing)
    inter_comm = MPI.Comm_split(comm, color, rank)

    # Leaders do the inter‑node sum
    global_leader = Vector{Float64}(undef, 2)
    if shm_rank == 0
        global_leader .= MPI.Allreduce(node_vals, MPI.SUM, inter_comm)
    end

    # Broadcast the final 3‑vector down to all threads on each node
    MPI.Bcast!(global_leader, 0, shm_comm)
    p2SAM, m2SAM = Tuple(global_leader)

    return (-log2(m2SAM/length(data(ψ))), 1.0 - p2SAM/length(data(ψ))) # TODO: should we really return 0.0 there?
end

# The refactored MC_quantity:
function MC_SRE(
    ψ,
    q,
    Nβ::Int,
    Nsamples::Int,
    seed::Union{Nothing,Int};
    cleanup::Bool = true,
)

    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Set a random seed (only on rank 0), then broadcast to all ranks
    seed = rank == 0 ? seed === nothing ? rand(1:(10^9)) : seed : 0
    seed = MPI.Bcast(seed, 0, comm)

    # Create a temporary directory for storing results
    tmpdir = rank == 0 ? mktempdir() : nothing
    tmpdir = MPI.bcast(tmpdir, 0, comm)

    # scatter the β‐indices
    idx_range, _ = scatter_range(Nβ, comm)
    β_vals = Float64.((idx_range .- 1) ./ (Nβ - 1))

    # do all the work inside the temp-dir context:
    m2 = 0.0
    try
        # each rank writes its chunk of files
        for (local_i, β) in enumerate(β_vals)
            global_idx = idx_range[1] + local_i - 1
            HadaMAG._compute_MC_SRE_β(
                ψ,
                q,
                Nsamples,
                seed + global_idx,
                β,
                global_idx,
                tmpdir,
            )
        end

        MPI.Barrier(comm)

        # rank 0 reads all the files, computes the Simpson integrals
        if rank == 0
            x, res_means, _, m2_means, _, _ =
                HadaMAG.process_files(seed; folder = tmpdir, Nβ)

            I_res = HadaMAG.integrate_simpson(x, res_means)
            I_m2ADD = HadaMAG.integrate_simpson(x, m2_means)

            m2 = -log2(2^(-I_res) + I_m2ADD)
        end

    finally
        if rank == 0 # Rank 0 cleans up the temporary directory
            if cleanup
                rm(tmpdir, force = true, recursive = true)
            else
                @info "Retaining temp directory at: $tmpdir"
            end
        end
    end

    MPI.Barrier(comm)

    # broadcast the final result m2 to all ranks
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
