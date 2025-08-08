using Printf
using FastHadamardStructuredTransforms_jll
using Libdl

"""
    MC_SRE2(ψ::StateVec{T,2}; backend = :auto, Nβ = 13, Nsamples = 1000, seed)

Compute the Stabilizer Renyi entropy (q=2) of a quantum state ψ using the Monte Carlo method (with Nsamples samples).
and the integral using Nβ points.

# Arguments
- `ψ`: A [`StateVec`](@ref) object representing the quantum state.
- `backend`: The backend to use for the computation. Default is `:auto`, which selects the best available backend.
- `Nβ`: Number of points for the integral. Default is 13.
- `Nsamples`: Number of samples for the Monte Carlo method. Default is 1000.
- `seed`: Random seed for reproducibility. Default is `nothing`, which uses a random seed.
"""
function MC_SRE2(
    ψ::StateVec{T,2};
    backend = :auto,
    Nβ = 13,
    Nsamples = 1000,
    seed = nothing,
) where {T}
    _apply_backend(_choose_backend(backend), :MC_SRE2, ψ, Nβ, Nsamples, seed)
end

function MC_SRE(
    ψ::StateVec{T,2},
    q::Number;
    backend = :auto,
    Nβ = 13,
    Nsamples = 1000,
    seed = nothing,
) where {T}
    _apply_backend(_choose_backend(backend), :MC_SRE, ψ, q, Nβ, Nsamples, seed)
end


# TODO: Check this description -> how is this new algorithm named?

"""
    SRE2(ψ::StateVec{T,2}; backend = :auto)

Compute the exact Stabilizer Renyi entropy (q=2) of a quantum state ψ using the HadaMAG algorithm.

# Arguments
- `ψ`: A [`StateVec`](@ref) object representing the quantum state.
- `backend`: The backend to use for the computation. Default is `:auto`, which selects the best available backend.
"""
function SRE2(ψ::StateVec{T,2}; backend = :auto) where {T}
    _apply_backend(_choose_backend(backend), :SRE2, ψ)
end

"""
    SRE(ψ::StateVec{T,2}; backend = :auto)

Compute the exact Stabilizer Renyi entropy (q) of a quantum state ψ using the HadaMAG algorithm.
Returns the SRE value and the lost norm of the state vector.

# Arguments
- `ψ`: A [`StateVec`](@ref) object representing the quantum state.
- `backend`: The backend to use for the computation. Default is `:auto`, which selects the best available backend.
"""
function SRE(ψ::StateVec{T,2}, q::Number; backend = :auto) where {T}
    _apply_backend(_choose_backend(backend), :SRE, ψ, q)
end

"""
    mana_SRE2(ψ::StateVec{T,3}; backend = :auto)

Compute the Mana of a quantum state qu-trit state ψ using the HadaMAG algorithm.
"""
function mana_SRE2(ψ::StateVec{T,3}; backend = :auto) where {T}
    _apply_backend(_choose_backend(backend), :mana_SRE2, ψ)
end

function _compute_MC_SRE2_β(
    ψ::StateVec{T,2},
    Nsamples::Int,
    seed::Union{Nothing,Int},
    β::Float64,
    j::Int,
    tmpdir::String,
) where {T}
    dim = length(data(ψ))
    L = qubits(ψ)
    pnorm = m2SAM = mval = 0.0

    rng = MersenneTwister(seed)
    dist = Uniform(0.0, 1.0)

    tmp1 = zeros(Float64, dim)
    tmp2 = zeros(Float64, dim)

    @inline compute_TMP!(tmp1, tmp2, data(ψ))

    # create arrays of real doubles that will store real and impaginary part of the state across the grays code
    inVR = zeros(Float64, dim)

    @inline pnorm, mval, m3val = sample_SRE2(L, data(ψ), tmp1, tmp2, inVR)

    # Batch RNG & tries outside the loop for performance
    randvals = rand(rng, dist, Nsamples)
    tries = floor.(Int, randvals .* 9) .+ 1 # each element ∈ 1:9
    rs = rand(rng, dist, Nsamples) # vector of random in [0,1)

    # Create a table of masks for each qubit position
    MASK_TABLE = UInt64.(1) .<< (0:(L-1))

    # Open file and init stats
    filename = "_$(j)_$(seed).dat"
    io = open(joinpath(tmpdir, filename), "w")
    buf = IOBuffer()
    sum_p = 0.0
    sum_p2 = 0.0
    n_p = 0
    currPROB = -Inf
    currM2 = 0.0
    currX = UInt64(0)

    FLUSH_INTERVAL = 1_000  # dump to file every FLUSH_INTERVAL iterations

    # Main loop over precomputed draws
    for idx = 1:Nsamples
        z = idx - 1 # 0:(Nsamples-1) index
        r = rs[idx] # pre‐drawn uniform
        tr = tries[idx] # 1…9

        # # generate tr random numbers of qubits to flip
        # whereA_vec = rand(rng, 1:L, tr)

        # # CHECK: foldl is better?
        # # build the XOR-mask by folding the table entries
        # # mask_acc   = foldl(⊻, MASK_TABLE[whereA_vec]; init = zero(UInt64))
        # mask_acc = zero(UInt64)
        # @inbounds for w in whereA_vec
        #     mask_acc ⊻= MASK_TABLE[w]
        # end

        mask_acc = zero(UInt64)
        @inbounds @simd for _ = 1:tr
            w = rand(rng, 1:L)          # single Int draw
            mask_acc ⊻= MASK_TABLE[w]
        end

        # apply mask to current state
        currX ⊻= mask_acc
        @inline apply_X_mask!(mask_acc, tmp1, tmp2)

        @inline pnorm, m2SAM = sample_SRE2(L, data(ψ), tmp1, tmp2, inVR)

        # accept/reject with only one exponentiation
        pβ = m2SAM^β
        if (r < pβ/currPROB) && (currX != 0)
            currPROB = pβ
            currM2 = -log2(m2SAM)
        else
            # undo mask
            currX ⊻= mask_acc

            # CHECK: parallelize applyX is it useful for L large?
            @inline apply_X_mask!(mask_acc, tmp1, tmp2)
        end

        # update stats
        sum_p += currM2
        sum_p2 += currM2^2
        n_p += 1

        # occasionally dump to disk
        ile = z > 1 ? 2^(Int(floor(log2(z)/1.5))) : 10
        if (z % ile == 0) && (n_p > 20)
            res = sum_p / n_p
            var = sum_p2 / n_p
            @printf(io, "%d %.20f %.20f %.20f\n", z, res, mval/dim, var)
        end

        # flush every FLUSH_INTERVAL log‐lines:
        # if buf.size ≥ FLUSH_INTERVAL * 60  # rough bytes estimate
        #     write(io, take!(buf)) # write + clear
        # end
    end

    write(io, take!(buf)) # final flush

    close(io)
end

function _compute_MC_SRE_β(
    ψ::StateVec{T,2},
    q::Number,
    Nsamples::Int,
    seed::Union{Nothing,Int},
    β::Float64,
    j::Int,
    tmpdir::String,
) where {T}
    dim = length(data(ψ))
    L = qubits(ψ)
    pnorm = mSAM = mval = 0.0

    rng = MersenneTwister(seed)
    dist = Uniform(0.0, 1.0)
    vq = Val(q) # Value type for q, used for method dispatch

    tmp1 = Vector{Float64}(undef, dim)
    tmp2 = Vector{Float64}(undef, dim)

    @inline compute_TMP!(tmp1, tmp2, data(ψ))

    # create arrays of real doubles that will store real and impaginary part of the state across the grays code
    inVR = zeros(Float64, dim)

    @inline pnorm, mval = sample_SRE(L, vq, data(ψ), tmp1, tmp2, inVR)

    # Batch RNG & tries outside the loop for performance
    randvals = rand(rng, dist, Nsamples)
    tries = floor.(Int, randvals .* 9) .+ 1 # each element ∈ 1:9
    rs = rand(rng, dist, Nsamples) # vector of random in [0,1)

    # Create a table of masks for each qubit position
    MASK_TABLE = UInt64.(1) .<< (0:(L-1))

    # Open file and init stats
    filename = "_$(j)_$(seed).dat"
    io = open(joinpath(tmpdir, filename), "w")
    buf = IOBuffer()
    sum_p = 0.0
    sum_p2 = 0.0
    n_p = 0
    currPROB = 1e-120
    currM2 = 0.0
    currX = UInt64(0)

    FLUSH_INTERVAL = 1_000  # dump to file every FLUSH_INTERVAL iterations

    # Main loop over precomputed draws
    for idx = 1:Nsamples
        z = idx - 1 # 0:(Nsamples-1) index
        r = rs[idx] # pre‐drawn uniform
        tr = tries[idx] # 1…9

        # generate tr random numbers of qubits to flip
        whereA_vec = rand(rng, 1:L, tr)

        # CHECK: foldl is better?
        # build the XOR-mask by folding the table entries
        # mask_acc   = foldl(⊻, BIT_MASK[whereA_vec]; init = zero(UInt64))
        mask_acc = zero(UInt64)
        @inbounds for w in whereA_vec
            mask_acc ⊻= MASK_TABLE[w]
        end

        # apply mask to current state
        currX ⊻= mask_acc
        @inline apply_X_mask!(mask_acc, tmp1, tmp2)

        @inline pnorm, mSAM = sample_SRE(L, vq, data(ψ), tmp1, tmp2, inVR)

        # accept/reject with only one exponentiation
        pβ = mSAM^β
        if (r < pβ/currPROB) && (currX != 0)
            currPROB = pβ
            currM2 = -log2(mSAM)
        else
            # undo mask
            currX ⊻= mask_acc

            # CHECK: parallelize applyX is it useful for L large?
            @inline apply_X_mask!(mask_acc, tmp1, tmp2)
        end

        # update stats
        sum_p += currM2
        sum_p2 += currM2^2
        n_p += 1

        # occasionally dump to disk
        ile = z > 1 ? 2^(Int(floor(log2(z)/1.5))) : 10
        if (z % ile == 0) && (n_p > 20)
            res = sum_p / n_p
            var = sum_p2 / n_p
            @printf(io, "%d %.20f %.20f %.20f\n", z, res, mval/dim, var)
        end

        # flush every FLUSH_INTERVAL log‐lines:
        # if buf.size ≥ FLUSH_INTERVAL * 60  # rough bytes estimate
        #     write(io, take!(buf)) # write + clear
        # end
    end

    write(io, take!(buf)) # final flush

    close(io)
end

@fastmath function sample_SRE2(
    L::Int,
    X::AbstractVector{ComplexF64},
    tmp1::AbstractVector{Float64},
    tmp2::AbstractVector{Float64},
    inVR::AbstractVector{Float64},
)
    @inline blend_fallback_muladd!(inVR, X, tmp1, tmp2)

    # Perform Fast Hadamard Transform
    call_fht!(inVR, Int32(L))

    @inline psam, m2sam = compute_moments(inVR)

    return psam / 2^L, m2sam
end

@fastmath function sample_SRE(
    L::Int,
    vq::Val,
    X::AbstractVector{ComplexF64},
    tmp1::AbstractVector{Float64},
    tmp2::AbstractVector{Float64},
    inVR::AbstractVector{Float64},
)
    @inline blend_fallback_muladd!(inVR, X, tmp1, tmp2)

    # Perform Fast Hadamard Transform
    call_fht!(inVR, Int32(L))

    @inline psam, msam = compute_moments(inVR, vq)

    return psam / 2^L, msam
end

@fastmath function compute_moments(inVR, ::Val{2})
    T = eltype(inVR)
    s2 = zero(T)
    s4 = zero(T)

    @inbounds @fastmath @simd for x in inVR
        y2 = x*x # x²

        s2 += y2 # plain sum of squares
        s4 = muladd(y2, y2, s4) # fuse y2*y2 + s4  →  s4 += x⁴
    end

    return s2, s4
end

@fastmath function compute_moments(inVR, ::Val{q}) where {q}
    T = eltype(inVR)
    s2 = zero(T)
    s4 = zero(T)

    @inbounds @fastmath @simd for x in inVR
        y2 = x*x # x²

        s2 += y2 # plain sum of squares
        s4 += y2 ^ q
    end

    return s2, s4
end

"""
    blend_fallback_muladd!(inVR::AbstractVector{Float64}, X::AbstractVector{ComplexF64}, TMP1::AbstractVector{Float64}, TMP2::AbstractVector{Float64})

Blend the real and imaginary parts of the complex vector X into inVR using SIMD operations.
The operation is performed in-place, modifying inVR directly.
"""
@fastmath function blend_fallback_muladd!(inVR, X, tmp1, tmp2)
    @inbounds for i in eachindex(inVR, X, tmp1, tmp2)
        inVR[i] = muladd(real(X[i]), tmp1[i], imag(X[i]) * tmp2[i])
    end
    return nothing
end

# """
#     compute_moments(inVR::AbstractVector{Float64})

# Compute the second, fourth, and sixth moments of the input vector inVR.
# The moments are computed using SIMD operations for performance.
# """
# @fastmath function compute_moments(inVR)
#     T = eltype(inVR)
#     s2 = zero(T)
#     s4 = zero(T)
#     s6 = zero(T)

#     @inbounds @fastmath @simd for x in inVR
#         # x²
#         y2 = x*x
#         # plain sum of squares
#         s2 += y2

#         # fuse y2*y2 + s4  →  s4 += x⁴
#         s4 = muladd(y2, y2, s4)

#         # fuse (x⁴)*x² + s6  →  s6 += x⁶
#         s6 = muladd(y2*y2, y2, s6)
#     end

#     return s2, s4, s6
# end

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

@fastmath @inline function apply_X_mask_2!(
    a::Vector{Float64},
    b::Vector{Float64},
    mask::UInt,
)
    dim = length(a)
    @inbounds for i = 1:dim
        # compute zero-based index once
        i0 = i - 1
        j = (i0 ⊻ mask) + 1
        if i < j
            # load
            ai = a[i];
            aj = a[j]
            bi = b[i];
            bj = b[j]
            # swap
            a[i] = aj;
            a[j] = ai
            b[i] = bj;
            b[j] = bi
        end
    end
    return a, b
end

"""
    compute_tmp!(tmp1::AbstractVector{T}, tmp2::AbstractVector{T}, X::AbstractVector{Complex{T}})

Compute the tmp1 and tmp2 vectors in-place using SIMD operations. tmp1 = real(X) + imag(X), tmp2 = imag(X) - real(X).
"""
@fastmath function compute_TMP!(
    tmp1::AbstractVector{T},
    tmp2::AbstractVector{T},
    X::AbstractVector{Complex{T}},
) where {T<:AbstractFloat}
    @inbounds @simd for i in eachindex(X)
        c = X[i]
        r = real(c)
        im = imag(c)
        tmp1[i] = r + im
        tmp2[i] = im - r
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

_compute_chunk_SRE2(
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
) = _compute_chunk_SRE2(0, istart, iend, ψ, Zwhere, XTAB)

@fastmath function _compute_chunk_SRE2(
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
)::Tuple{Float64,Float64,Float64}
    L = qubits(ψ)
    dim = 2^L

    p2SAM = m2SAM = m3SAM = 0.0

    TMP1 = zeros(Float64, dim)
    TMP2 = zeros(Float64, dim)
    Xloc1 = zeros(Float64, dim)
    Xloc2 = zeros(Float64, dim)

    for i = 1:dim
        r, im = real(ψ[i]), imag(ψ[i])
        Xloc1[i] = r
        Xloc2[i] = im
        TMP1[i] = r + im
        TMP2[i] = im - r
    end

    # create arrays of real doubles that will store real and impaginary part of the state across the Gray's code
    inVR = zeros(Float64, dim)

    # for given Pauli string at position istart, specified in XTAB, act with appropriate X_j operators
    for el = 1:L
        bits = XTAB[istart]
        bitval = (bits >> (el - 1)) & 0x1 # el runs from 1, 2, 3,... we need (el-1) in Julia, so that el=1 picks out the least-significant bit.
        if bitval == 1
            @inline apply_X!(el - 1, TMP1, TMP2) # Act with the X operators that are 1 in Gray's code
        end
    end

    # the worker will update the state TMP when going through the greys code form istart to iend
    for ix = istart:iend
        # non-trivial mathematical thing happening: we need to compute such a vector related to the
        # state (Xloc) and its propagated version along the Gray's code, TMP
        blend_muladd!(inVR, Xloc1, TMP1, Xloc2, TMP2)  # inVR[r] = Xloc1[r] * TMP1[r] + Xloc2[r] * TMP2[r] in an FMA (fused multiply-add) way

        # Do the fast Hadamard transform of the inVR
        call_fht!(inVR, Int32(L))

        # the vectors obtained with FHT contain overlaps of given Pauli strings:
        # the Pauli strings are of the form XTAB[ix] (0...1 corresponding to Z operator)
        # so to calculate SRE we have to add entries of the resulting vector with the specified powers
        # (depending on the index of SRE we calculate)
        #
        # this step does sum over all Z pauli strings, given their X part determined by XTAB[ix]
        # in time complexity L*2^L (while the naive implementation would be 4^L)
        #
        # in order to calculate SRE
        for r = 1:dim
            # p = copy(inVR[r])
            pnorm = inVR[r] ^ 2 #+ inVI[r] * inVI[r]
            m2SAM += pnorm ^ 2 # CHECK: Int exponent is faster?
            # m3SAM += pnorm ^ 3 # TODO: should we uncomment this line?
        end

        # this takes us from Pauli string at given position of the Greys code to the next one
        # (by a single action of X_j operator )
        if ix + index < dim
            @inline apply_X!(Zwhere[ix] - 1, TMP1, TMP2)
        end

    end

    return p2SAM, m2SAM, m3SAM
end

_compute_chunk_SRE(
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    q::Number,
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
) = _compute_chunk_SRE(Val(q), 0, istart, iend, ψ, Zwhere, XTAB)

_compute_chunk_SRE(
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    q::Number,
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
) = _compute_chunk_SRE(Val(q), index, istart, iend, ψ, Zwhere, XTAB)

@fastmath function _compute_chunk_SRE(
    ::Val{q},
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
)::Tuple{Float64,Float64} where {q}
    L = qubits(ψ)
    dim = 2^L

    p2SAM = mSAM = 0.0

    TMP1 = zeros(Float64, dim)
    TMP2 = zeros(Float64, dim)
    Xloc1 = zeros(Float64, dim)
    Xloc2 = zeros(Float64, dim)

    for i = 1:dim
        r, im = real(ψ[i]), imag(ψ[i])
        Xloc1[i] = r
        Xloc2[i] = im
        TMP1[i] = r + im
        TMP2[i] = im - r
    end

    # create arrays of real doubles that will store real and impaginary part of the state across the Gray's code
    inVR = zeros(Float64, dim)

    # for given Pauli string at position istart, specified in XTAB, act with appropriate X_j operators
    for el = 1:L
        bits = XTAB[istart]
        bitval = (bits >> (el - 1)) & 0x1 # el runs from 1, 2, 3,... we need (el-1) in Julia, so that el=1 picks out the least-significant bit.
        if bitval == 1
            @inline apply_X!(el - 1, TMP1, TMP2) # Act with the X operators that are 1 in Gray's code
        end
    end

    # the worker will update the state TMP when going through the greys code form istart to iend
    for ix = istart:iend
        # we compute such a vector related to the state (Xloc) and its propagated version along the Gray's code (TMP)
        blend_muladd!(inVR, Xloc1, TMP1, Xloc2, TMP2)  # inVR[r] = Xloc1[r] * TMP1[r] + Xloc2[r] * TMP2[r] in an FMA (fused multiply-add) way

        # Do the fast Hadamard transform of the inVR
        call_fht!(inVR, Int32(L))

        # the vectors obtained with FHT contain overlaps of given Pauli strings:
        # the Pauli strings are of the form XTAB[ix] (0...1 corresponding to Z operator)
        # so to calculate SRE we have to add entries of the resulting vector with the specified powers
        # (depending on the index of SRE we calculate)
        for r = 1:dim
            pnorm = inVR[r] ^ 2 #+ inVI[r] * inVI[r]
            p2SAM += pnorm
            mSAM += pnorm ^ q
        end

        # this takes us from Pauli string at given position of the Greys code to the next one
        # (by a single action of X_j operator )
        if ix + index < dim
            @inline apply_X!(Zwhere[ix] - 1, TMP1, TMP2)
        end
    end

    return p2SAM, mSAM
end

_compute_chunk_SRE_v2(
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
) = _compute_chunk_SRE_v2(Val(q), 0, istart, iend, ψ, Zwhere, XTAB, TMP1, TMP2, Xloc1, Xloc2)

_compute_chunk_SRE_v2(
    index::Int64,
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
) = _compute_chunk_SRE_v2(Val(q), index, istart, iend, ψ, Zwhere, XTAB, TMP1, TMP2, Xloc1, Xloc2)

# Here we do the work unitl "for ix = istart:iend" before and then mpi
@fastmath function _compute_chunk_SRE_v2(
    ::Val{q},
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
    TMP1::Vector{Float64},
    TMP2::Vector{Float64},
    Xloc1::Vector{Float64},
    Xloc2::Vector{Float64},
)::Tuple{Float64,Float64} where {q}
    L = qubits(ψ)
    dim = 2^L

    p2SAM = mSAM = 0.0

    # create arrays of real doubles that will store real and impaginary part of the state across the Gray's code
    inVR = zeros(Float64, dim)

    # for given Pauli string at position istart, specified in XTAB, act with appropriate X_j operators
    for el = 1:L
        bits = XTAB[istart]
        bitval = (bits >> (el - 1)) & 0x1 # el runs from 1, 2, 3,... we need (el-1) in Julia, so that el=1 picks out the least-significant bit.
        if bitval == 1
            @inline apply_X!(el - 1, TMP1, TMP2) # Act with the X operators that are 1 in Gray's code
        end
    end

    # the worker will update the state TMP when going through the greys code form istart to iend
    for ix = istart:iend
        # we compute such a vector related to the state (Xloc) and its propagated version along the Gray's code (TMP)
        blend_muladd!(inVR, Xloc1, TMP1, Xloc2, TMP2)  # inVR[r] = Xloc1[r] * TMP1[r] + Xloc2[r] * TMP2[r] in an FMA (fused multiply-add) way

        # Do the fast Hadamard transform of the inVR
        call_fht!(inVR, Int32(L))

        # the vectors obtained with FHT contain overlaps of given Pauli strings:
        # the Pauli strings are of the form XTAB[ix] (0...1 corresponding to Z operator)
        # so to calculate SRE we have to add entries of the resulting vector with the specified powers
        # (depending on the index of SRE we calculate)
        for r = 1:dim
            pnorm = inVR[r] ^ 2 #+ inVI[r] * inVI[r]
            p2SAM += pnor
            mSAM += pnorm ^ q
        end

        # this takes us from Pauli string at given position of the Greys code to the next one
        # (by a single action of X_j operator )
        if ix + index < dim
            @inline apply_X!(Zwhere[ix] - 1, TMP1, TMP2)
        end
    end

    return p2SAM, mSAM
end

# Here we do the work unitl "for ix = istart:iend" before and then mpi
@fastmath function _compute_chunk_SRE_v2(
    ::Val{2},
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
    TMP1::Vector{Float64},
    TMP2::Vector{Float64},
    Xloc1::Vector{Float64},
    Xloc2::Vector{Float64},
)::Tuple{Float64,Float64}
    L = qubits(ψ)
    dim = 2^L

    p2SAM = m2SAM = 0.0

    # create arrays of real doubles that will store real and impaginary part of the state across the Gray's code
    inVR = zeros(Float64, dim)

    # for given Pauli string at position istart, specified in XTAB, act with appropriate X_j operators
    for el = 1:L
        bits = XTAB[istart]
        bitval = (bits >> (el - 1)) & 0x1 # el runs from 1, 2, 3,... we need (el-1) in Julia, so that el=1 picks out the least-significant bit.
        if bitval == 1
            @inline apply_X!(el - 1, TMP1, TMP2) # Act with the X operators that are 1 in Gray's code
        end
    end

    # the worker will update the state TMP when going through the greys code form istart to iend
    for ix = istart:iend
        # we compute such a vector related to the state (Xloc) and its propagated version along the Gray's code (TMP)
        blend_muladd!(inVR, Xloc1, TMP1, Xloc2, TMP2)  # inVR[r] = Xloc1[r] * TMP1[r] + Xloc2[r] * TMP2[r] in an FMA (fused multiply-add) way

        # Do the fast Hadamard transform of the inVR
        call_fht!(inVR, Int32(L))

        # the vectors obtained with FHT contain overlaps of given Pauli strings:
        # the Pauli strings are of the form XTAB[ix] (0...1 corresponding to Z operator)
        # so to calculate SRE we have to add entries of the resulting vector with the specified powers
        # (depending on the index of SRE we calculate)
        for r = 1:dim
            pnorm = inVR[r] ^ 2 #+ inVI[r] * inVI[r]
            p2SAM += pnorm
            m2SAM += pnorm ^ 2
        end

        # this takes us from Pauli string at given position of the Greys code to the next one
        # (by a single action of X_j operator )
        if ix + index < dim
            @inline apply_X!(Zwhere[ix] - 1, TMP1, TMP2)
        end
    end

    return p2SAM, m2SAM
end

_compute_chunk_SRE_v23(
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
) = _compute_chunk_SRE_v23(Val(q), 0, istart, iend, ψ, Zwhere, XTAB, TMP1, TMP2, Xloc1, Xloc2)

_compute_chunk_SRE_v23(
    index::Int64,
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
) = _compute_chunk_SRE_v23(Val(q), index, istart, iend, ψ, Zwhere, XTAB, TMP1, TMP2, Xloc1, Xloc2)

@fastmath function _compute_chunk_SRE_v23(
    ::Val{q},
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
    TMP1::Vector{Float64},
    TMP2::Vector{Float64},
    Xloc1::Vector{Float64},
    Xloc2::Vector{Float64},
)::Tuple{Float64,Float64} where {q}
    L = qubits(ψ)
    dim = 2^L

    p2SAM = mSAM = 0.0

    L   = qubits(ψ)
    dim = 1<<L
    inVR = zeros(Float64, dim)

    # Loop over this thread’s chunk in Gray-code order:
    @inbounds for ix = istart:(iend)
        anypassed = false

        if ix == istart
            # initial setup (step 0 of Gray code): apply all bits in XTAB[istart]
            for site in 0:(L-1)
                if ((XTAB[istart] >> site) & 0x1) == 1
                    anypassed = true
                    # swap+blend for that site
                    @inline flip_and_update!(site, dim, TMP1, TMP2, Xloc1, Xloc2, inVR)
                end
            end

            if !anypassed
                # if no bit was passed, we just blend the vectors
                blend_muladd!(inVR, Xloc1, TMP1, Xloc2, TMP2) # inVR[r] = Xloc1[r] * TMP1[r] + Xloc2[r] * TMP2[r] in an FMA (fused multiply-add) way
            end
        else
            # advance one Gray‐code step by flipping bit Zwhere[ix-1]
            if ix - 1 + index < dim
                site = Zwhere[ix-1] - 1
                @inline flip_and_update!(site, dim, TMP1, TMP2, Xloc1, Xloc2, inVR)
            end
        end

        # --- now do the big transform + reduction
        call_fht!(inVR, Int32(L))

        @inbounds @simd for r in 1:dim
            pnorm = inVR[r] ^ 2
            p2SAM += pnorm
            mSAM += pnorm ^ q
        end
    end

    return p2SAM, mSAM
end

@fastmath function _compute_chunk_SRE_v23(
    ::Val{2},
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
    TMP1::Vector{Float64},
    TMP2::Vector{Float64},
    Xloc1::Vector{Float64},
    Xloc2::Vector{Float64},
)::Tuple{Float64,Float64}
    L = qubits(ψ)
    dim = 2^L

    p2SAM = m2SAM = 0.0

    L   = qubits(ψ)
    dim = 1<<L
    inVR = zeros(Float64, dim)

    m2SAM = 0.0

    # Loop over this thread’s chunk in Gray-code order:
    for ix = istart:(iend)
        anypassed = false
        if ix == istart
            # --- initial setup (step 0 of Gray code): apply all bits in XTAB[istart]
            for site in 0:(L-1)
                if ((XTAB[istart] >> site) & 0x1) == 1
                    anypassed = true
                    # swap+blend for that site
                    @inline flip_and_update!(site, dim, TMP1, TMP2, Xloc1, Xloc2, inVR)
                end
            end

            if !anypassed
                # if no bit was passed, we just blend the vectors
                @inline blend_muladd!(inVR, Xloc1, TMP1, Xloc2, TMP2)  # inVR[r] = Xloc1[r] * TMP1[r] + Xloc2[r] * TMP2[r] in an FMA (fused multiply-add) way
            end
        else
            # --- advance one Gray‐code step by flipping bit Zwhere[ix-1]
            if ix - 1 + index < dim
                site = Zwhere[ix-1] - 1
                @inline flip_and_update!(site, dim, TMP1, TMP2, Xloc1, Xloc2, inVR)
            end
        end

        @inline call_fht!(inVR, Int32(L))

        @inbounds @simd for r in 1:dim
            pnorm = inVR[r] ^ 2
            p2SAM += pnorm
            m2SAM += pnorm ^ 2
        end
    end

    return p2SAM, m2SAM
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
    site::Int, dim::Int,
    TMP1::AbstractVector{T}, TMP2::AbstractVector{T},
    X1::AbstractVector{T},  X2::AbstractVector{T},
    inVR::AbstractVector{T}
) where {T<:AbstractFloat}
    stride     = 1 << site
    period2    = stride << 1
    half_pairs = dim >>> 1

    @inbounds @simd for k in 0:(half_pairs-1)
        # compute indices
        block  = k >>> site
        offset = k & (stride - 1)
        base   = block * period2
        i      = base + offset + 1
        j      = i + stride

        # swap in TMP1/TMP2
        t1i = TMP1[i]; t1j = TMP1[j]
        TMP1[i] = t1j; TMP1[j] = t1i

        t2i = TMP2[i]; t2j = TMP2[j]
        TMP2[i] = t2j; TMP2[j] = t2i

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
    Threads.@threads for idx in 0:(4^n - 1)
        # decode idx into base‐4 “digits” p[1],…,p[n] in {0,1,2,3}
        tmp = idx
        P = Paulis[(tmp % 4) + 1]
        tmp ÷= 4
        # build the full n‐qubit operator P = P₁ ⊗ P₂ ⊗ … ⊗ Pₙ
        for _ in 2:n
            P = kron(P, Paulis[(tmp % 4) + 1])
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
    for idx in 0:(4^n - 1)
        tmp = idx
        # build P = P₁⊗…⊗Pₙ by decoding idx in base-4
        P = Paulis[(tmp % 4) + 1];  tmp ÷= 4
        for _ in 2:n
            P = kron(P, Paulis[(tmp % 4) + 1])
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
