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
function SRE2(ψ::StateVec{T,2}; backend = :auto, progress = true, kwargs...) where {T}
    _apply_backend(_choose_backend(backend), :SRE, ψ, 2; progress, kwargs...)
end

"""
    SRE(ψ::StateVec{T,2}; backend = :auto)

Compute the exact Stabilizer Renyi entropy (q) of a quantum state `ψ` using the HadaMAG algorithm.
Returns the SRE value and the lost norm of the state vector.

# Arguments
- `ψ`: A [`StateVec`](@ref) object representing the quantum state.
- `q`: The Renyi index (the most common value is 2).

# Keyword Arguments
- `backend`: The backend to use for the computation. Default is `:auto`, which selects the best available backend.
- `progress`: Whether to show a progress bar. Default to `true`.
"""
function SRE(
    ψ::StateVec{T,2},
    q::Number;
    backend = :auto,
    progress = true,
    kwargs...,
) where {T}
    _apply_backend(_choose_backend(backend), :SRE, ψ, q; progress, kwargs...)
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
            j = Int(UInt64(i-1) ⊻ mask) + 1
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

    sM2 = 0.0
    sM2sq = 0.0
    sP2 = 0.0
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

@fastmath @inline function kahan_accumulate(::Val{2}, inVR)::Tuple{Float64,Float64}
    p2_sum = 0.0
    p2_c = 0.0
    m2_sum = 0.0
    m2_c = 0.0

    @inbounds for v in inVR
        p = v * v
        # Kahan for p2SAM
        y = p - p2_c
        t = p2_sum + y
        p2_c = (t - p2_sum) - y
        p2_sum = t

        pp = p * p
        # Kahan for mSAM
        y2 = pp - m2_c
        t2 = m2_sum + y2
        m2_c = (t2 - m2_sum) - y2
        m2_sum = t2
    end

    return (p2_sum, m2_sum)
end

@fastmath @inline function kahan_accumulate(
    ::Val{q},
    inVR,
)::Tuple{Float64,Float64} where {q}
    p2_sum = 0.0
    p2_c = 0.0 # compensation for p2_sum

    m_sum = 0.0
    m_c = 0.0 # compensation for m_sum

    @inbounds for v in inVR
        p = v * v
        pq = p^q

        # accumulate p into p2_sum with compensation
        y = p - p2_c
        t = p2_sum + y
        p2_c = (t - p2_sum) - y
        p2_sum = t

        # accumulate pq into m_sum with compensation
        y2 = pq - m_c
        t2 = m_sum + y2
        m_c = (t2 - m_sum) - y2
        m_sum = t2
    end

    return (p2_sum, m_sum)
end

compute_chunk_sre(
    istart::Int64,
    iend::Int64,
    ψ::StateVec{ComplexF64,2},
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
    ψ::StateVec{ComplexF64,2},
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

function compute_chunk_sre(
    ::Val{q}, # exponent (Int or Real)
    istart::Int,
    iend::Int,
    ψ::StateVec{ComplexF64,2},
    Zwhere::Vector{Int},
    XTAB::Vector{UInt64},
    TMP1::Vector{Float64}, # shared, read-only
    TMP2::Vector{Float64}, # shared, read-only
    X1::Vector{Float64}, # shared, read-only
    X2::Vector{Float64}, # shared, read-only
    inVR::Vector{Float64}, # thread-local scratch, length == dim
    pbar::AbstractProgress,
    stride::Int,
)::Tuple{Float64,Float64} where {q}
    L = qubits(ψ)
    dim = 1 << L
    L32 = Int32(L)

    # Outer Kahan accumulators
    p2SAM = 0.0
    mSAM = 0.0

    mask = UInt(0)
    cnt = 0

    @inbounds for ix = istart:iend
        # cheap progress tick every `stride` (still commented out)
        if stride > 0
            cnt += 1
            if (cnt % stride) == 0
                tick!(pbar, stride)
                cnt = 0
            end
        end

        if ix == istart
            bits = XTAB[ix]
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
            site = Zwhere[ix-1] - 1  # your data is 1-based sites
            mask = flip_and_update_mask!(site, dim, mask, TMP1, TMP2, X1, X2, inVR)
        end

        # Walsh-Hadamard / FHT on inVR
        call_fht!(inVR, L32)

        # Kahan over inVR for p2SAM and mSAM
        local_p2, local_m = kahan_accumulate(Val(q), inVR)

        # Accumulate p2 and mSAM
        p2SAM += local_p2
        mSAM += local_m
    end

    finish!(pbar)

    return (p2SAM, mSAM)
end

"""
    compute_chunk_sre(::Val{2}, istart, iend, ψ, Zwhere, XTAB, TMP1, TMP2, X1, X2, inVR) -> (p2, m2)

Compute contributions for items `ix ∈ [istart, iend]` using a per-thread scratch `inVR`.

Key performance points:
- `TMP1/TMP2/X1/X2` are shared, *read-only* across threads.
- `inVR` is *thread-local* and reused (no allocations in the hot path).
- Flips are applied via XOR index masking (no in-place swapping of TMPs).
- For q=2 we use `p*p` (no `^`), and `dim = 1 << L` (cheap/int-exact).
"""
function compute_chunk_sre(
    ::Val{2},
    istart::Int,
    iend::Int,
    ψ::StateVec{ComplexF64,2},
    Zwhere::Vector{Int},
    XTAB::Vector{UInt64},
    TMP1::Vector{Float64}, # shared, read-only
    TMP2::Vector{Float64}, # shared, read-only
    X1::Vector{Float64}, # shared, read-only
    X2::Vector{Float64}, # shared, read-only
    inVR::Vector{Float64}, # thread-local scratch
    pbar::AbstractProgress,
    stride::Int,
)::Tuple{Float64,Float64}
    L = qubits(ψ)
    dim = 1 << L
    L32 = Int32(L)

    # Outer Kahan accumulators
    p2SAM = 0.0
    mSAM = 0.0

    mask = UInt(0)
    cnt = 0
    @inbounds for ix = istart:iend
        # cheap progress tick every `stride` (still commented out)
        if stride > 0
            cnt += 1
            if (cnt % stride) == 0
                tick!(pbar, stride)
                cnt = 0
            end
        end

        if ix == istart
            bits = XTAB[ix]
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
            site = Zwhere[ix-1] - 1  # your data is 1-based sites
            mask = flip_and_update_mask!(site, dim, mask, TMP1, TMP2, X1, X2, inVR)
        end

        # Walsh-Hadamard / FHT on inVR
        call_fht!(inVR, L32)

        # Kahan over inVR for p2SAM and mSAM
        local_p2, local_m = kahan_accumulate(Val(2), inVR)

        # Accumulate p2 and mSAM
        p2SAM += local_p2
        mSAM += local_m
    end

    finish!(pbar)

    return (p2SAM, mSAM)
end

"""
    flip_and_update_mask!(site, dim, mask, TMP1, TMP2, X1, X2, inVR) -> newmask

Apply a single-bit flip at `site` (0-based) on top of the existing `mask`.
We never mutate `TMP1/TMP2`, instead we read from indices XORed by the
new mask and write the blended values into `inVR`.

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
        idx_i = (i - 1) ⊻ newmask + 1 # source for i
        idx_j = (j - 1) ⊻ newmask + 1 # source for j

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
    iphase(k::Int) -> ComplexF64

Compute i^k for k mod 4 without allocations:
    k%4 == 0→1, 1→i, 2→-1, 3→-i.
Used for the global phase i^{|x∧z|} from Y operators.
"""
@inline @fastmath function iphase(k::Int)::ComplexF64
    k2 = k & 0x3
    k2 == 0 && return 1.0 + 0im
    k2 == 1 && return 0.0 + 1im
    k2 == 2 && return -1.0 + 0im
    return 0.0 - 1im
end

# choose an unsigned type for the masks (x,z)
@inline function masktype(n::Int)
    n <= 32 && return UInt32
    n <= 64 && return UInt64
    error(
        "n=$n not supported by the bitmask method (needs >64-bit masks)... are you sure you want to compute this for that many qubits?",
    )
end

"""
    expval_pauli_mask(ψ, x, z) -> Float64

Expectation value ⟨ψ|P|ψ⟩ for an n-qubit Pauli string P using *bitmask encoding*.

Pauli strings are encoded with two n-bit masks (x, z) where, for each qubit j:
    (x_j, z_j) = (0,0) → I
                  (1,0) → X
                  (0,1) → Z
                  (1,1) → Y (= i XZ)

Key identity (action on computational basis |b⟩ with b an n-bit index):
    P_{x,z} |b⟩ = i^{|x∧z|} (-1)^{b⋅z} | b ⊻ x ⟩
where:
    ∧  = bitwise AND,
    ⊻  = bitwise XOR,
    |x∧z| = popcount(x & z) = number of Y’s,
    b⋅z   = parity(popcount(b & z)).

Therefore
    ⟨ψ|P_{x,z}|ψ⟩ = i^{|x∧z|}  ∑_b  (-1)^{b⋅z}  conj(ψ_{b⊻x}) ψ_b

This function evaluates that sum in O(2^n) time and O(1) extra memory,
without ever building the 2^n×2^n Pauli matrix.
"""
@inline function expval_pauli_mask(ψ::Vector{ComplexF64}, x::UInt32, z::UInt32)::Float64
    N = length(ψ) # state size (N = 2^n)
    ph = iphase(count_ones(x & z)) # i^{#Y} global phase (Y iff x&z has a 1)
    s = 0.0 + 0.0im # complex accumulator for the sum

    @inbounds @simd for b = 0:(N-1) # loop over all basis indices b
        bb = UInt(b) # use UInt for cheap bit ops
        bp = bb ⊻ x # b' = b XOR x  (flip X/Y positions)

        # (-1)^{b⋅z}: parity of bits where both b and z are 1
        sign = ifelse(isodd(count_ones(bb & z)), -1.0, 1.0)
        # accumulate conj(ψ[b']) * ψ[b] * (-1)^{b⋅z}
        s += conj(ψ[Int(bp)+1]) * ψ[b+1] * sign
    end

    # result is mathematically real; take real to drop roundoff imag parts
    return real(ph * s)
end

@inline function expval_pauli_mask(ψ::Vector{ComplexF64}, x::UInt64, z::UInt64)::Float64
    N = length(ψ) # state size (N = 2^n)
    ph = iphase(count_ones(x & z)) # i^{#Y} global phase (Y iff x&z has a 1)
    s = 0.0 + 0.0im # complex accumulator for the sum

    @inbounds @simd for b = 0:(N-1) # loop over all basis indices b
        bb = UInt64(b) # use UInt for cheap bit ops
        bp = bb ⊻ x # b' = b XOR x  (flip X/Y positions)

        # (-1)^{b⋅z}: parity of bits where both b and z are 1
        sign = ifelse(isodd(count_ones(bb & z)), -1.0, 1.0)
        # accumulate conj(ψ[b']) * ψ[b] * (-1)^{b⋅z}
        s += conj(ψ[Int(bp)+1]) * ψ[b+1] * sign
    end

    # result is mathematically real; take real to drop roundoff imag parts
    return real(ph * s)
end

naive_SRE(ψ::StateVec{ComplexF64,2}, q::Integer; progress = true) =
    naive_SREnaive_SRE_kahan2(data(ψ), q; progress = progress)

"""
    naive_SRE(ψ, q)

Naive stabilizer-Rényi entropy of integer order `q` for an n-qubit pure state ψ, it uses a
 compensated summation algorithm for improved numerical stability.
"""
@fastmath function naive_SRE(ψ::Vector{<:Complex}, q::Integer; progress = true)
    N = length(ψ)
    @assert ispow2(N) "length(ψ) = $N is not 2^n"
    n = Int(trailing_zeros(UInt(N)))
    # T = masktype(n)  # x,z mask type (up to 64 qubits - like you would ever need that much...)

    @assert n <= 32 "n=$n too large for bitmask method (max 32 qubits supported)... Are you sure you want to compute this for that many qubits?"

    total = Int(1) << (2*n) # 4^n

    pbar = progress ? HadaMAG.CounterProgress(total; hz = 8) : HadaMAG.NoProgress()
    progress_stride = progress ? max(div(total, 100), 1) : 0
    cnt = 0

    partial = zeros(Float64, Threads.nthreads())
    c_val = zeros(Float64, Threads.nthreads()) # compensation for each thread

    Threads.@threads for idx = 0:(total-1)
        if progress_stride > 0
            cnt += 1
            if (cnt % progress_stride) == 0
                tick!(pbar, progress_stride)
                cnt = 0
            end
        end

        tid = Threads.threadid()

        # decode idx (0 ... 4^n-1) as an n-qubit Pauli string in base-4:
        #  0 → I, 1 → X, 2 → Y, 3 → Z
        tmp = UInt32(idx)
        x = UInt32(0)
        z = UInt32(0)

        @inbounds for j = 0:(n-1)
            d = tmp & 0x3 # next base-4 digit
            if d == 0x1 # X  → (x,z) = (1,0)
                x |= UInt32(1) << j
            elseif d == 0x2 # Y  → (x,z) = (1,1)
                x |= UInt32(1) << j
                z |= UInt32(1) << j
            elseif d == 0x3 # Z  → (x,z) = (0,1)
                z |= UInt32(1) << j
            end
            tmp >>>= 2 # move to next digit
        end

        # e = ⟨ψ|P_{x,z}|ψ⟩ for this Pauli string
        e = expval_pauli_mask(ψ, x, z)

        # Compensated summation of (e^2)^q into per-thread accumulator
        y = (e*e)^q - c_val[tid]
        t = partial[tid] + y
        c_val[tid] = (t - partial[tid]) - y
        partial[tid] = t
    end

    finish!(pbar)
    return -log2(sum(partial) / N)
end
