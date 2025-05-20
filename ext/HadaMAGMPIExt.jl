module HadaMAGMPIExt

using HadaMAG
using MPI

# This module provides the low-level kernels for the MPI backend.

function MC_SRE2(ψ, Nβ::Int, Nsamples::Int, seed::Union{Nothing,Int}; cleanup = true)
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
    for j in 1:mpisize
        start = Int(ceil((j - 1) * Nβ / mpisize)) + 1
        stop = Int(ceil(j * Nβ / mpisize))
        push!(beta_values, beta_val.(range_beta[start:stop]))
    end

    beta_rank = beta_values[rank+1]

    m2 = 0.0
    try
        # Each rank processes its own beta values
        for i in 1:length(beta_rank)
            β = beta_rank[i]
        # for i = 1:Nβ
            # β = Float64(i) / Nβ
            # convert beta value to j
            idx = (β * (Nβ - 1) |> round |> Int) + 1

            HadaMAG._compute_SRE2_β(ψ, Nsamples, seed + idx, β, idx, tmpdir)
        end

        MPI.Barrier(MPI.COMM_WORLD)

        # Rank 0 computes the average of the results for each β
        if rank == 0
            x, res_means, res_stds, m2ADD_means, m2ADD_stds =
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

end