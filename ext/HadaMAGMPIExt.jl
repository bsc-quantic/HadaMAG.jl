module HadaMAGMPIExt

using HadaMAG
using MPI

# This module provides the low-level kernels for the MPI backend.

function MC_SRE2(
    ψ,
    Nβ::Int,
    Nsamples::Int,
    seed::Union{Nothing,Int};
    cleanup = true,
)
    # Set a random seed
    seed = seed === nothing ? floor(Int, rand() * 1e9) : seed
    tmpdir = mktempdir()

    # TODO: create a macro for this mpi rank assignment
    MPI.Initialized() || MPI.Init()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    comm = MPI.COMM_WORLD
    mpisize = MPI.Comm_size(comm)

    range_beta = 1:Nβ
    beta_val(i) = Float64(i - 1) / (Nβ - 1) # so that β ∈ [0, 1]

    beta_values = []
    for j = 1:mpisize
        start = Int(ceil((j - 1) * Nβ / mpisize)) + 1
        stop = Int(ceil(j * Nβ / mpisize))
        push!(beta_values, beta_val.(range_beta[start:stop]))
    end

    beta_rank = beta_values[rank+1]

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


function SRE2(ψ)
    dim = length(data(ψ))
    L = qubits(ψ)
    Ncores = Threads.nthreads() # Number of threads per mpi rank

    # TODO: create a macro for this mpi rank assignment
    MPI.Initialized() || MPI.Init()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)

    p2SAM = m2SAM = m3SAM = 0.0

    XTAB, Zwhere = generate_gray_table(L, 2)

    # Here we decide how many elements each process should get
    base = div(dim - 1, nprocs)
    rem = mod(dim - 1, nprocs)

    # Process i (0-indexed) gets base+1 if i < rem, else base:
    counts = [i < rem ? base+1 : base for i = 0:(nprocs-1)]

    # Compute displacements (starting index in the global array for each process)
    displs = [sum(counts[1:i]) for i = 0:(nprocs-1)]

    # Determine how many elements the current process gets
    local_count = counts[rank+1]  # note: Julia uses 1-indexing


    base_xtab = div(dim, nprocs)
    rem_xtab = mod(dim, nprocs)
    counts_xtab = [i < rem_xtab ? base_xtab+1 : base_xtab for i = 0:(nprocs-1)]
    displs_xtab = [sum(counts_xtab[1:i]) for i = 0:(nprocs-1)]

    # Create a local array to hold the slice
    if rank == 0
        sender_zwhere = VBuffer(Zwhere, counts)
        sender_xtab = VBuffer(XTAB, counts_xtab)
    else
        sender_zwhere = VBuffer(nothing)
        sender_xtab = VBuffer(nothing)
    end

    MPI.Barrier(comm)

    # Scatter the appropriate slice of Zwhere
    local_Zwhere = MPI.Scatterv!(sender_zwhere, zeros(Int64, local_count), 0, comm)
    local_XTAB = MPI.Scatterv!(sender_xtab, zeros(UInt64, counts_xtab[rank+1]), 0, comm)

    # TODO: Wrap this in a function?
    # divide the gray's code (which has 2^N positions) into Ncores, more or less equal patches
    # we store the starting and finishing index corresponding to each of the patches in istart and iend
    istart = [div((i-1)*(length(local_Zwhere)), Ncores) + 1 for i = 1:Ncores]
    iend = [div(i*(length(local_Zwhere)), Ncores) for i = 1:Ncores]

    if rank == MPI.Comm_size(comm) - 1
        iend[end] += 1
    end

    # we compute starting_id, which is the index of the first element of the local_Zwhere array, in the global Zwhere array
    starting_id = displs[rank+1]

    # The last thread processes until the end of Z (here we mimic Z.size()+1 from C++ by adding one extra element, if needed)
    PS2 = zeros(Float64, Ncores)
    MS2 = zeros(Float64, Ncores)
    MS3 = zeros(Float64, Ncores)

    Threads.@threads for j = 1:Ncores
        # Each thread/processes the subrange [istart[j], iend[j]-1]
        ps2, ms2, ms3 =
            HadaMAG._compute_chunk_SRE(starting_id, istart[j], iend[j], ψ, Zwhere, XTAB)
        PS2[j] = ps2
        MS2[j] = ms2
        MS3[j] = ms3
    end

    # Now sum the partial sums to obtain the final results.
    p2SAM = sum(PS2)
    m2SAM = sum(MS2)
    m3SAM = sum(MS3)

    # Sum all the partial results of each rank on the master rank
    if rank == 0
        p2SAM = MPI.Reduce(p2SAM, MPI.SUM, 0, comm)
        m2SAM = MPI.Reduce(m2SAM, MPI.SUM, 0, comm)
        m3SAM = MPI.Reduce(m3SAM, MPI.SUM, 0, comm)
    else
        MPI.Reduce(p2SAM, MPI.SUM, 0, comm)
        MPI.Reduce(m2SAM, MPI.SUM, 0, comm)
        MPI.Reduce(m3SAM, MPI.SUM, 0, comm)
    end

    return (-log2(m2SAM/dim), 0.0) # TODO: should we really return 0.0 there?
end

end
