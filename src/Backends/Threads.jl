module ThreadedBackend
using LinearAlgebra
using Random
using HadaMAG

# This module provides the low-level kernels for the Threaded backend.

function MC_SRE2(ψ, Nβ::Int, Nsamples::Int, seed::Union{Nothing,Int}; cleanup = true)
    # Set a random seed
    seed = seed === nothing ? floor(Int, rand() * 1e9) : seed
    tmpdir = mktempdir()

    m2 = 0.0
    try
        Threads.@threads for i = 1:Nβ
            β = Float64(i) / Nβ

            HadaMAG._compute_SRE2_β(ψ, Nsamples, seed + i, β, i, tmpdir)
        end

        # Compute the average of the results for each β
        x, res_means, res_stds, m2ADD_means, m2ADD_stds =
            HadaMAG.process_files(seed; folder = tmpdir, Nβ)

        # Compute the final result using Simpson's rule
        integral_res = HadaMAG.integrate_simpson(x, res_means)
        integral_m2ADD = HadaMAG.integrate_simpson(x, m2ADD_means)

        m2 = -log2(2^(-integral_res) + integral_m2ADD)
    finally
        if cleanup
            rm(tmpdir, force = true, recursive = true)
        else
            @info "Retaining temp directory at: $tmpdir"
        end
    end

    return m2
end

end # module ThreadedBackend
