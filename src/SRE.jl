using Printf
using FastHadamardStructuredTransforms_jll
using Libdl

function MC_SRE2(
    ψ::StateVec{T,2};
    backend = :auto,
    Nβ = 13,
    Nsamples = 1000,
    seed = nothing,
    progress = true,
) where {T}
    _apply_backend(_choose_backend(backend), :MC_SRE, ψ, 2, Nβ, Nsamples; seed, progress)
end

"""
    MC_SRE(ψ::StateVec{T,2}; backend = :auto, Nβ = 13, Nsamples = 1000, seed)

Compute the Stabilizer Renyi entropy (q=2) of a quantum state ψ using the Monte Carlo method (with Nsamples samples).
and the integral using Nβ points.

# Arguments
- `ψ`: A [`StateVec`](@ref) object representing the quantum state.
- `q`: The Renyi index (the most common value is 2).

# Keyword Arguments
- `backend`: The backend to use for the computation. Default to `:auto`, which selects the best available backend.
- `Nβ`: Number of points for the integral. Default to 13.
- `Nsamples`: Number of samples for the Monte Carlo method. Default to 1000.
- `seed`: Random seed for reproducibility. Default to `nothing`, which uses a random seed.
- `progress`: Whether to show a progress bar. Default to `true`.
"""
function MC_SRE(
    ψ::StateVec{T,2},
    q::Number;
    backend = :auto,
    Nβ = 13,
    Nsamples = 1000,
    seed = nothing,
    progress = true,
) where {T}
    _apply_backend(_choose_backend(backend), :MC_SRE, ψ, q, Nβ, Nsamples; seed, progress)
end


"""
    SRE2(ψ::StateVec{T,2}; backend = :auto, progress = true)

Compute the exact Stabilizer Renyi entropy (q=2) of a quantum state ψ using the HadaMAG algorithm.

# Arguments
- `ψ`: A [`StateVec`](@ref) object representing the quantum state.

# Keyword Arguments
- `backend`: The backend to use for the computation. Default is `:auto`, which selects the best available backend.
- `progress`: Whether to show a progress bar. Default to `true`.
"""
function SRE2(ψ::StateVec{T,2}; backend = :auto, progress = true) where {T}
    _apply_backend(_choose_backend(backend), :SRE, ψ, 2; progress)
end

"""
    SRE(ψ::StateVec{T,2}; backend = :auto)

Compute the exact Stabilizer Renyi entropy (q) of a quantum state ψ using the HadaMAG algorithm.
Returns the SRE value and the lost norm of the state vector.

# Arguments
- `ψ`: A [`StateVec`](@ref) object representing the quantum state.
- `q`: The Renyi index (the most common value is 2).

# Keyword Arguments
- `backend`: The backend to use for the computation. Default is `:auto`, which selects the best available backend.
- `progress`: Whether to show a progress bar. Default to `true`.
"""
function SRE(ψ::StateVec{T,2}, q::Number; backend = :auto, progress = true) where {T}
    _apply_backend(_choose_backend(backend), :SRE, ψ, q; progress)
end

"""
    mana_SRE2(ψ::StateVec{T,3}; backend = :auto)

Compute the Mana of a quantum state qu-trit state ψ using the HadaMAG algorithm.
"""
function mana_SRE2(ψ::StateVec{T,3}; backend = :auto) where {T}
    _apply_backend(_choose_backend(backend), :mana_SRE2, ψ)
end

"""
    mc_sre_β!(X, tmp1, tmp2, inVR, q, Nsamples, seed, β, L)
      -> (⟨M2⟩, ⟨M2²⟩, m2ADD, ⟨pnorm⟩, n)

Compute the Monte Carlo estimate at a single β with Metropolis sampling.

Buffers:
- `tmp1/tmp2` are shared, read-only (built once with `build_tmp!`),
- `inVR` is thread-local scratch (reused across calls),
- `X` is the complex state (read-only).

No heap allocations in the hot loop.
"""
mc_sre_β!(
    X::AbstractVector{ComplexF64},
    tmp1::AbstractVector{Float64},
    tmp2::AbstractVector{Float64},
    inVR::AbstractVector{Float64},
    q::Number,
    Nsamples::Int,
    seed::Int,
    β::Float64,
    L::Int,
) = mc_sre_β!(X, tmp1, tmp2, inVR, q, Nsamples, seed, β, L, NoProgress(), 0)

@fastmath function mc_sre_β!(
    X::AbstractVector{ComplexF64},
    tmp1::AbstractVector{Float64},
    tmp2::AbstractVector{Float64},
    inVR::AbstractVector{Float64},
    q::Number,
    Nsamples::Int,
    seed::Int,
    β::Float64,
    L::Int,
    pbar::AbstractProgress,
    stride::Int,
)
    dim = length(X)
    L32 = Int32(L)

    # masked blend (no swaps)
    @inline @fastmath function blend_mask!(mask::UInt64)
        @inbounds @simd for i = 1:dim
            j = Int(UInt(i-1) ⊻ mask) + 1
            c = X[i]
            inVR[i] = muladd(real(c), tmp1[j], imag(c) * tmp2[j])
        end
        nothing
    end
    @inline function sample_mask!(mask::UInt64)
        blend_mask!(mask)
        call_fht!(inVR, L32)
        psam, msam = HadaMAG.compute_moments(inVR, Val(q))
        return (psam / dim, msam) # pnorm, m_q
    end

    rng = MersenneTwister(seed)
    randvals = rand(rng, Nsamples)
    tries = floor.(Int, randvals .* 9) .+ 1 # 1..9
    rs = rand(rng, Nsamples)

    p2_0, m0 = sample_mask!(UInt64(0))
    MASK_TBL = UInt64.(1) .<< (0:(L-1))

    currMask = UInt64(0)
    currPROB = 1e-120
    currM2 = 0.0
    currP2 = p2_0

    sM2 = 0.0;
    sM2sq = 0.0;
    sP2 = 0.0;
    n = 0
    @inbounds for t = 1:Nsamples
        if stride > 0 && (t % stride) == 0
            tick!(pbar, stride)
        end

        tr = tries[t]
        propΔ = zero(UInt64)
        @inbounds for _ = 1:tr
            site = rand(rng, 1:L)
            propΔ ⊻= MASK_TBL[site]
        end
        propMask = currMask ⊻ propΔ

        p2_prop, m_prop = sample_mask!(propMask)
        pβ = m_prop^β
        r = rs[t]

        if (r < pβ / currPROB) && (propMask != 0)
            currMask = propMask
            currPROB = pβ
            currM2 = -log2(m_prop)
            currP2 = p2_prop
        end

        sM2 += currM2
        sM2sq += currM2 * currM2
        sP2 += currP2
        n += 1
    end

    finish!(pbar)

    return sM2/n, sM2sq/n, m0/dim, sP2/n, n
end

# Specialized method for q=2
@fastmath function compute_moments(inVR, ::Val{2})
    T = eltype(inVR)
    s2 = zero(T)
    s4 = zero(T)

    @inbounds @fastmath @simd for x in inVR
        y2 = x*x
        s2 += y2 # plain sum of squares

        s4 = muladd(y2, y2, s4) # fuse y2*y2 + s4 -> s4 += x⁴
    end

    return s2, s4
end

@fastmath function compute_moments(inVR, ::Val{q}) where {q}
    T = eltype(inVR)
    s2 = zero(T)
    s4 = zero(T)

    @inbounds @fastmath @simd for x in inVR
        y2 = x*x
        s2 += y2 # plain sum of squares

        s4 += y2 ^ q
    end

    return s2, s4
end

# TODO: Test apply_X! multithreaded
"""
    apply_X!(site, Xs...)

Apply the “flip-bit at `site`” X-operator in-place
to each vector in `Xs`.  All vectors must have the same length.
"""
@fastmath function apply_X!(site::Int, Xs::AbstractVector{T}...) where {T}
    dim = length(Xs[1])
    mask = 1 << site # No need to subtract 1, since `site` is 0-based
    @inbounds for i = 1:dim
        j = ((i-1) ⊻ mask) + 1
        if j > i
            for X in Xs
                X[i], X[j] = X[j], X[i]
            end
        end
    end
    return nothing
end

"""
    apply_X_mask!(mask::UInt, Xs::AbstractVector{T}...) where {T}

Apply the multi-bit X-mask `mask` in-place to every vector in `Xs` (we assume that all have the same length).
For each 1-based index `i`, compute the zero-based partner index `j = ((i-1) ⊻ mask) + 1`;
whenever `i < j`, swap `a[i] ↔ a[j]` and `b[i] ↔ b[j]`.

Returns the modified `(a, b)`.
This is equivalent to performing all single‐bit flips whose bit‐positions are set in `mask` in one pass.
"""
@fastmath @inline function apply_X_mask!(mask::UInt, Xs::AbstractVector{T}...) where {T}
    dim = length(Xs[1])
    @inbounds for i = 1:dim
        # compute zero-based index once
        i0 = i - 1
        j = (i0 ⊻ mask) + 1
        if i < j
            for X in Xs
                X[i], X[j] = X[j], X[i]
            end
        end
    end
    return nothing
end

# TODO: add a comment or fix the description here
# inVR[r] = Xloc1[r] * TMP1[r] + Xloc2[r] * TMP2[r] in an FMA (fused multiply-add) way
# This is a SIMD-friendly version of the above operation.
@inline @fastmath function blend_muladd!(
    inVR::AbstractVector{T},
    X1::AbstractVector{T},
    T1::AbstractVector{T},
    X2::AbstractVector{T},
    T2::AbstractVector{T},
) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(inVR, X1, T1, X2, T2)
        # fuse one multiply+add into an FMA:
        inVR[i] = muladd(X1[i], T1[i], X2[i] * T2[i])
    end
    return nothing
end

compute_chunk_sre(
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    q::Number,
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
    TMP1::Vector{Float64},
    TMP2::Vector{Float64},
    Xloc1::Vector{Float64},
    Xloc2::Vector{Float64},
    inVR::Vector{Float64},
) = compute_chunk_sre(
    Val(q),
    istart,
    iend,
    ψ,
    Zwhere,
    XTAB,
    TMP1,
    TMP2,
    Xloc1,
    Xloc2,
    inVR,
    NoProgress(),
    0,
)

compute_chunk_sre(
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    q::Number,
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
    TMP1::Vector{Float64},
    TMP2::Vector{Float64},
    Xloc1::Vector{Float64},
    Xloc2::Vector{Float64},
    inVR::Vector{Float64},
    pbar::AbstractProgress,
    stride::Int,
) = compute_chunk_sre(
    Val(q),
    istart,
    iend,
    ψ,
    Zwhere,
    XTAB,
    TMP1,
    TMP2,
    Xloc1,
    Xloc2,
    inVR,
    pbar,
    stride,
)

@fastmath function compute_chunk_sre(
    ::Val{q},                        # exponent (Int or Real)
    istart::Int,
    iend::Int,
    ψ::StateVec{Float64,2},
    lvec1::Vector{Int},               # your original `local_vector1`
    lvec2::Vector{UInt64},            # your original `local_vector2` (bitmasks)
    TMP1::Vector{Float64},            # shared, read-only
    TMP2::Vector{Float64},            # shared, read-only
    X1::Vector{Float64},              # shared, read-only
    X2::Vector{Float64},              # shared, read-only
    inVR::Vector{Float64},            # thread-local scratch, length == dim
    pbar::AbstractProgress,
    stride::Int,
)::Tuple{Float64,Float64} where {q}
    L = qubits(ψ)
    dim = 1 << L

    p2SAM = 0.0
    mSAM = 0.0
    L32 = Int32(L)

    mask = UInt(0)
    cnt = 0

    @inbounds for ix = istart:iend
        # cheap progress tick every `stride`
        if stride > 0
            cnt += 1
            if (cnt % stride) == 0
                tick!(pbar, stride)
                cnt = 0
            end
        end

        if ix == istart
            bits = lvec2[ix]
            if bits == 0
                # No flips: one fused blend
                @simd for r = 1:dim
                    inVR[r] = muladd(X1[r], TMP1[r], X2[r]*TMP2[r])
                end
            else
                # Apply all flips encoded in `bits`
                # (iterate set bits: trailing_zeros / clear-lowest-set-bit)
                m = bits
                while m != 0
                    site = Int(trailing_zeros(m))         # 0-based site
                    mask = flip_and_update_mask!(site, dim, mask, TMP1, TMP2, X1, X2, inVR)
                    m &= m - 1
                end
            end
        else
            # Subsequent ix: flip a single site relative to previous mask
            site = lvec1[ix-1] - 1                          # your data is 1-based sites
            mask = flip_and_update_mask!(site, dim, mask, TMP1, TMP2, X1, X2, inVR)
        end

        # Walsh-Hadamard / FHT on inVR
        call_fht!(inVR, L32)

        @inbounds @simd for r = 1:dim
            p = inVR[r]^2
            p2SAM += p
            mSAM += p^q
        end
    end

    finish!(pbar)

    return (p2SAM, mSAM)
end

"""
    compute_chunk_sre(::Val{2}, istart, iend, ψ, lvec1, lvec2, TMP1, TMP2, X1, X2, inVR) -> (p2, m2)

Compute contributions for items `ix ∈ [istart, iend]` using a per-thread scratch `inVR`.

Key performance points:
- `TMP1/TMP2/X1/X2` are shared, *read-only* across threads.
- `inVR` is *thread-local* and reused (no allocations in the hot path).
- Flips are applied via XOR index masking (no in-place swapping of TMPs).
- For q=2 we use `p*p` (no `^`), and `dim = 1 << L` (cheap/int-exact).
"""
@fastmath function compute_chunk_sre(
    ::Val{2},                        # exponent (Int or Real)
    istart::Int,
    iend::Int,
    ψ::StateVec{Float64,2},
    lvec1::Vector{Int},               # your original `local_vector1`
    lvec2::Vector{UInt64},            # your original `local_vector2` (bitmasks)
    TMP1::Vector{Float64},            # shared, read-only
    TMP2::Vector{Float64},            # shared, read-only
    X1::Vector{Float64},              # shared, read-only
    X2::Vector{Float64},              # shared, read-only
    inVR::Vector{Float64},            # thread-local scratch, length == dim
    pbar::AbstractProgress,
    stride::Int,
)::Tuple{Float64,Float64}
    L = qubits(ψ)
    dim = 1 << L

    p2SAM = 0.0
    mSAM = 0.0
    L32 = Int32(L)

    mask = UInt(0)
    cnt = 0

    @inbounds for ix = istart:iend
        # cheap progress tick every `stride`
        if stride > 0
            cnt += 1
            if (cnt % stride) == 0
                tick!(pbar, stride)
                cnt = 0
            end
        end

        if ix == istart
            bits = lvec2[ix]
            if bits == 0
                # No flips: one fused blend
                @simd for r = 1:dim
                    inVR[r] = muladd(X1[r], TMP1[r], X2[r]*TMP2[r])
                end
            else
                # Apply all flips encoded in `bits`
                # (iterate set bits: trailing_zeros / clear-lowest-set-bit)
                m = bits
                while m != 0
                    site = Int(trailing_zeros(m)) # 0-based site
                    mask = flip_and_update_mask!(site, dim, mask, TMP1, TMP2, X1, X2, inVR)
                    m &= m - 1
                end
            end
        else
            # Subsequent ix: flip a single site relative to previous mask
            site = lvec1[ix-1] - 1 # your data is 1-based sites
            mask = flip_and_update_mask!(site, dim, mask, TMP1, TMP2, X1, X2, inVR)
        end

        # Walsh-Hadamard / FHT on inVR
        call_fht!(inVR, L32)

        @inbounds @simd for r = 1:dim
            p = inVR[r]^2
            p2SAM += p
            mSAM += p * p
        end
    end

    finish!(pbar)

    return (p2SAM, mSAM)
end

"""
    flip_and_update_mask!(site, dim, mask, TMP1, TMP2, X1, X2, inVR) -> newmask

Apply a single-bit flip at `site` (0-based) on top of the existing `mask`.
We never mutate `TMP1/TMP2`, instead we read from indices XORed by the
*new* mask and write the blended values into `inVR`.

- `TMP1/TMP2/X1/X2` are shared, read-only.
- `inVR` is thread-local; written in full each call.
- Returns the updated mask (`mask ⊻ (1 << site)`).
"""
@inline @fastmath function flip_and_update_mask!(
    site::Int,
    dim::Int,
    mask::UInt,
    TMP1::AbstractVector{T},
    TMP2::AbstractVector{T},
    X1::AbstractVector{T},
    X2::AbstractVector{T},
    inVR::AbstractVector{T},
)::UInt where {T<:AbstractFloat}
    stride = 1 << site
    period2 = stride << 1
    halfpairs = dim >>> 1
    newmask = mask ⊻ UInt(stride)

    @inbounds @simd for k = 0:(halfpairs-1)
        # Choose (i,j) so adding `stride` never carries into higher bits.
        block = k >>> site
        offs = k & (stride - 1)
        base = block * period2
        i = base + offs + 1
        j = i + stride

        # Map to original positions under the *new* mask
        idx_i = (i - 1) ⊻ newmask + 1    # source for i
        idx_j = (j - 1) ⊻ newmask + 1    # source for j

        inVR[i] = muladd(X1[i], TMP1[idx_i], X2[i] * TMP2[idx_i])
        inVR[j] = muladd(X1[j], TMP1[idx_j], X2[j] * TMP2[idx_j])
    end
    return newmask
end

"""
    flip_and_update!(
        site::Int, dim::Int,
        TMP1::AbstractVector{T}, TMP2::AbstractVector{T},
        X1::AbstractVector{T},  X2::AbstractVector{T},
        inVR::AbstractVector{T}
    ) where T<:AbstractFloat

Perform the in-place “butterfly” swap + FMA updates on `TMP1`, `TMP2`, and `inVR`
corresponding to flipping the single bit at position `site`.
Executes exactly `dim/2` iterations (no branches), computing each pair of indices
on the fly so that the loop is fully SIMD-vectorizable.
"""
@inline @fastmath function flip_and_update!(
    site::Int,
    dim::Int,
    TMP1::AbstractVector{T},
    TMP2::AbstractVector{T},
    X1::AbstractVector{T},
    X2::AbstractVector{T},
    inVR::AbstractVector{T},
) where {T<:AbstractFloat}
    stride = 1 << site
    period2 = stride << 1
    half_pairs = dim >>> 1

    @inbounds @simd for k = 0:(half_pairs-1)
        # compute indices
        block = k >>> site
        offset = k & (stride - 1)
        base = block * period2
        i = base + offset + 1
        j = i + stride

        # swap in TMP1/TMP2
        t1i = TMP1[i];
        t1j = TMP1[j]
        TMP1[i] = t1j;
        TMP1[j] = t1i

        t2i = TMP2[i];
        t2j = TMP2[j]
        TMP2[i] = t2j;
        TMP2[j] = t2i

        # fused multiply‐add update
        inVR[i] = muladd(X1[i], t1j, X2[i] * t2j)
        inVR[j] = muladd(X1[j], t1i, X2[j] * t2i)
    end
    return nothing
end

"""
    actX_qutrit!(ψ::AbstractVector{Complex{T}}, site::Integer) where T

Apply the qutrit X‐gate on `site` (1‐based, least‐significant digit = 1) to the
state‐vector `ψ` of length 3^n, in place.  This cycles each local triple

  |…0ᵢ…⟩→|…1ᵢ…⟩→|…2ᵢ…⟩→|…0ᵢ…⟩

on the chosen ternary digit, with no extra allocations.

Throws an error if `length(ψ)` isn’t a power of 3 or `site` ∉ 1:n.
"""
@fastmath function actX_qutrit!(ψ::Vector{Complex{T}}, site::Integer) where {T}
    N = length(ψ)
    n = Int(round(log(N)/log(3)))
    N != 3^n && error("length(ψ) = $N is not 3^n")
    # site ∉ 1:n && error("site=$(site) out of 0:$(n-1)")

    stride = 3^(site-1) # We subtract one since site is 1-based
    block = 3*stride

    @inbounds for block_start = 1:block:N
        @simd for offset = 0:(stride-1)
            a = block_start + offset
            b = a + stride
            c = b + stride

            # Cycle the three states at positions a, b, c
            tmp = ψ[a]
            ψ[a] = ψ[b]
            ψ[b] = ψ[c]
            ψ[c] = tmp
        end
    end

    return ψ
end

# Do the naive 4^N implementation of SRE2, which is not efficient but serves as a reference.
function naive_SRE2(ψ::Vector{ComplexF64})
    N = length(ψ)
    # check that N is a power of two, set n = # qubits
    n = Int(round(log2(N)))
    2^n != N && error("length(ψ) = $N is not 2^n")

    # define the single‐qubit Pauli matrices
    σI = [1.0 0.0; 0.0 1.0]
    σX = [0.0 1.0; 1.0 0.0]
    σY = [0.0 -im; im 0.0]
    σZ = [1.0 0.0; 0.0 -1.0]
    Paulis = (σI, σX, σY, σZ)

    acc = 0.0
    # loop over all 4^n Pauli strings
    Threads.@threads for idx = 0:(4^n-1)
        # decode idx into base‐4 “digits” p[1],…,p[n] in {0,1,2,3}
        tmp = idx
        P = Paulis[(tmp%4)+1]
        tmp ÷= 4
        # build the full n‐qubit operator P = P₁ ⊗ P₂ ⊗ … ⊗ Pₙ
        for _ = 2:n
            P = kron(P, Paulis[(tmp%4)+1])
            tmp ÷= 4
        end
        # expectation value ⟨ψ|P|ψ⟩ is real for Hermitian P (up to numerical noise)
        exp_val = real(ψ' * (P * ψ))
        acc += exp_val^4
    end

    # normalize by 2^n and take −log₂
    return -log2(acc / N)
end

"""
    naive_SRE(ψ::Vector{Complex{T}}, k::Integer) where {T}

Compute the “naive” stabilizer-Rényi entropy of integer order `k` for an n-qubit pure state ψ
(of length 2ⁿ).
"""
function naive_SRE(ψ::Vector{Complex{T}}, k::Integer) where {T<:Real}
    N = length(ψ)
    n = Int(round(log2(N)))
    2^n != N && error("length(ψ) = $N is not 2^n")
    # single-qubit Paulis
    σI = [1.0 0.0; 0.0 1.0]
    σX = [0.0 1.0; 1.0 0.0]
    σY = [0.0 -im; im 0.0]
    σZ = [1.0 0.0; 0.0 -1.0]
    Paulis = (σI, σX, σY, σZ)

    acc = 0.0
    # loop over all 4^n Pauli strings
    for idx = 0:(4^n-1)
        tmp = idx
        # build P = P₁⊗…⊗Pₙ by decoding idx in base-4
        P = Paulis[(tmp%4)+1];
        tmp ÷= 4
        for _ = 2:n
            P = kron(P, Paulis[(tmp%4)+1])
            tmp ÷= 4
        end

        # expectation ⟨ψ|P|ψ⟩ (should be real up to numerical noise)
        e = real(ψ' * (P * ψ))
        acc += e^(2*k)
    end

    # final SRE_k = −log₂[(1/2^n) Σ e^(2k)]
    return -log2(acc / N)
end

@fastmath function actX_qutrit!(ψ::Vector{Complex{T}}, site::Integer, k::Integer) where {T}
    N = length(ψ)
    n = Int(round(log(N)/log(3)))
    N != 3^n && error("length(ψ) = $N is not 3^n")
    # site ∉ 1:n     && error("site=$site out of 1:$n")

    stride = 3^(site-1)
    block = 3*stride

    @inbounds for bs = 1:block:N
        @simd for off = 0:(stride-1)
            i0 = bs + off
            i1 = i0 + stride
            i2 = i1 + stride

            if k == 1
                # forward cycle: 0→1→2→0
                ψ[i0], ψ[i1], ψ[i2] = ψ[i1], ψ[i2], ψ[i0]
            elseif k == 2
                # backward cycle: 0→2→1→0 (≡ X²)
                ψ[i0], ψ[i1], ψ[i2] = ψ[i2], ψ[i0], ψ[i1]
            end
        end
    end

    return ψ
end

_compute_chunk_mana_SRE(
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,3},
    Zwhere::Vector{Int64},
    XTAB::Matrix{Int64},
) = _compute_chunk_mana_SRE(0, istart, iend, ψ, Zwhere, XTAB)

@fastmath function _compute_chunk_mana_SRE(
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,3},
    Zwhere::Vector{Int64},
    XTAB::Matrix{Int64},
)::Tuple{Float64,Float64,ComplexF64}
    L = qudits(ψ)
    dim = 3^L

    p2SAM = m2SAM = m3SAM = 0.0

    TMP = copy(data(ψ))
    Xloc = copy(data(ψ))
    inV = zeros(ComplexF64, dim)
    @assert size(XTAB, 2) == 3^L

    for site = 1:L
        k = XTAB[site, istart]
        k > 0 && actX_qutrit!(TMP, site, k)
    end

    # the worker will update the state TMP when going through the greys code form istart to iend
    for ix = istart:iend
        # non-trivial mathematical thing happening: I need to calculate such a vector related to the
        # state (Xloc) and its propagated version along the grays code, TMP
        for r = 1:dim
            inV[r] = conj(Xloc[r]) * copy(TMP[r])
        end

        # We do fast Hadamard transform of the inVR, inVI
        @inline fast_hadamard_qutrit!(inV)

        # the vectors obtained with FHT  contain overlaps of given Pauli strings:
        # the Pauli strings are of the form XTAB[ix] (0...1 corresponding to Z operator)
        # so to calculate SRE we have to add entries of the resulting vector with the specified powers
        # (depending on the index of SRE we calculate)
        # this step does sum over all Z pauli strings, given their X part determined by XTAB[ix]
        # in time complexity L*2^L (while the naive implementation would be 4^L)

        @inbounds @simd for i in eachindex(inV)
            p = inV[i]
            ap = abs(p)              # |p|
            p2SAM += ap

            m2 = abs2(p)             # |p|^2
            m2SAM += m2 * m2             # |p|^4

            p2 = p * p
            m3SAM += p2 * p             # p^3
        end

        # this takes us from Pauli string at given position of the Greys code to the next one
        # (by a single action of X_j operator )
        if ix + index < dim
            @inline actX_qutrit!(TMP, Zwhere[ix])
        end

    end

    return p2SAM, m2SAM, m3SAM
end
