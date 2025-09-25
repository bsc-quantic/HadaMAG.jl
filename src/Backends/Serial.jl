module SerialBackend # This module provides the low-level kernels for the Serial backend.

using LinearAlgebra
using Random
using HadaMAG

function SRE(ψ, q; progress = true)
    dim = length(data(ψ))
    L = qubits(ψ)

    XTAB, Zwhere = generate_gray_table(L, 2)

    # Allocate the *full* scratch arrays…
    TMP1 = Vector{Float64}(undef, dim)
    TMP2 = Vector{Float64}(undef, dim)
    Xloc1 = Vector{Float64}(undef, dim)
    Xloc2 = Vector{Float64}(undef, dim)

    # … and initialize them in parallel
    Threads.@threads for i = 1:dim
        r = real(ψ[i])
        im = imag(ψ[i])
        Xloc1[i] = r
        Xloc2[i] = im
        TMP1[i] = r + im
        TMP2[i] = im - r
    end

    inVR = Vector{Float64}(undef, dim)

    pbar = progress ? HadaMAG.CounterProgress(length(XTAB); hz = 8) : HadaMAG.NoProgress()
    progress_stride = progress ? max(div(length(XTAB), 100), 10) : 0 # update ~100 times

    p2SAM, mSAM = HadaMAG.compute_chunk_sre(
        1,
        dim,
        ψ,
        q,
        Zwhere,
        XTAB,
        TMP1,
        TMP2,
        Xloc1,
        Xloc2,
        inVR,
        pbar,
        progress_stride,
    )

    return (-log2(mSAM/dim), abs(1-p2SAM/dim))
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

    # Precompute tmp1/tmp2
    tmp1 = Vector{Float64}(undef, dim)
    tmp2 = Vector{Float64}(undef, dim)
    HadaMAG.build_tmp!(tmp1, tmp2, X)

    # Per-thread scratch for inVR
    inVR = Vector{Float64}(undef, dim)

    # Outputs per β
    M2 = Vector{Float64}(undef, Nβ)
    M2ADD = Vector{Float64}(undef, Nβ)
    P2 = Vector{Float64}(undef, Nβ)

    base_seed = seed === nothing ? rand(1:(10^9)) : seed

    p =
        progress ? HadaMAG.CounterProgress(Nsamples; hz = 8, io = stderr) :
        HadaMAG.NoProgress()
    progress_stride = progress ? max(div(Nsamples, 100), 10) : 0 # update ~100 times

    for i = 1:Nβ
        β = (i - 1) / (Nβ - 1)
        m2_mean, _m2sq, m2add_mean, p2_mean, _n = HadaMAG.mc_sre_β!(
            X,
            tmp1,
            tmp2,
            inVR,
            q,
            Nsamples,
            base_seed + i,
            β,
            L,
            p,
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

function Mana(ψ; progress = true)
    dim = length(data(ψ))
    L = qudits(ψ)

    XTAB, Zwhere = generate_gray_table(L, 3)

    # Allocate all the necessary arrays
    TMP = Vector{ComplexF64}(undef, dim)
    conj_Xloc = Vector{ComplexF64}(undef, dim)

    for i = 1:dim
        TMP[i] = ψ[i]
        conj_Xloc[i] = conj(ψ[i])
    end

    inV = zeros(ComplexF64, dim)

    pbar = progress ? HadaMAG.CounterProgress(size(XTAB, 2); hz = 8) : HadaMAG.NoProgress()
    progress_stride = progress ? max(div(size(XTAB, 2), 100), 10) : 0 # update ~100 times

    p2SAM = HadaMAG.compute_chunk_mana_qutrits(1, dim, ψ, Zwhere, XTAB, TMP, conj_Xloc, inV, pbar, progress_stride)

    return log2(p2SAM)/log2(3.0)
end

end # module SerialBackend
