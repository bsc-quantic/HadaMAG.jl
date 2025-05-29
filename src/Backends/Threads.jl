module ThreadedBackend # This module provides the low-level kernels for the Threaded backend

using LinearAlgebra
using Random
using HadaMAG

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
        Threads.@threads for i = 1:Nβ
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
    Ncores = Threads.nthreads() # Number of threads

    p2SAM = m2SAM = m3SAM = 0.0

    XTAB, Zwhere = generate_gray_table(L, 2)

    # TODO: Wrap this in a function?
    # divide the gray's code (which has 2^N positions) into Ncores, more or less equal patches
    # we store the starting and finishing index corresponding to each of the patches in istart and iend
    istart = [div((i-1)*(length(Zwhere)+1), Ncores) + 1 for i = 1:Ncores]
    iend = [div(i*(length(Zwhere)+1), Ncores) for i = 1:Ncores]

    # The last thread processes until the end of Z (here we mimic Z.size()+1 from C++ by adding one extra element, if needed)
    PS2 = zeros(Float64, Ncores)
    MS2 = zeros(Float64, Ncores)
    MS3 = zeros(Float64, Ncores)

    Threads.@threads for j = 1:Ncores
        # Each thread/processes the subrange [istart[j], iend[j]-1]
        ps2, ms2, ms3 = HadaMAG._compute_chunk_SRE(istart[j], iend[j], ψ, Zwhere, XTAB)
        PS2[j] = ps2
        MS2[j] = ms2
        MS3[j] = ms3
    end

    # Now sum the partial sums to obtain the final results.
    p2SAM = sum(PS2)
    m2SAM = sum(MS2)
    m3SAM = sum(MS3)

    return (-log2(m2SAM/dim), 0.0) # TODO: should we really return 0.0 there?
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

    # SRE is returned here
    return (-log2(m2SAM/dim)/log2(3.0), -1.0*log2(real(m3SAM)/dim)/log2(3.0))
end

end # module ThreadedBackend
