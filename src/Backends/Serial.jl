module SerialBackend
using LinearAlgebra
using Random
using HadaMAG

# This module provides the low-level kernels for the Serial backend.

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

    m2 = 0.0
    try
        for i = 1:Nβ
            β = Float64(i - 1) / (Nβ - 1) # so that β ∈ [0, 1]

            HadaMAG._compute_MC_SRE2_β(ψ, Nsamples, seed + i, β, i, tmpdir)
        end

        # Compute the average of the results for each β
        x, res_means, res_stds, m2ADD_means, m2ADD_stds, naccepted =
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

function SRE2(ψ)
    dim = length(data(ψ))
    L = qubits(ψ)

    Zwhere = zeros(Int64, dim - 1)
    VALS = zeros(Int64, dim)
    XTAB = zeros(UInt64, dim)

    prev = 0
    for i = 1:((1<<L)-1)
        # Compute Gray code: val = i ^ (i >> 1)
        # In Julia, ⊻ (typed \xor<tab>) is bitwise xor
        val = i ⊻ (i >> 1)
        diff = val ⊻ prev

        VALS[i+1] = val
        prev = val

        # Convert to UInt64, i.e. treat as a 64-bit mask.
        r = UInt64(val)
        pr = UInt64(diff)
        XTAB[i+1] = r

        # Julia provides fast functions to count leading or trailing zeros in a bitset
        # Use trailing_zeros to find the index of the least-significant set bit.
        # (For example, for r = 8 the result is 3, because 8 == 0b1000.)
        where_ = trailing_zeros(pr)
        Zwhere[i] = where_
    end

    p2SAM, m2SAM, m3SAM = HadaMAG._compute_chunk_SRE(1, dim, ψ, Zwhere, XTAB)

    return (-log2(m2SAM/dim), 0.0) # TODO: should we really return 0.0 there?
end

end # module SerialBackend
