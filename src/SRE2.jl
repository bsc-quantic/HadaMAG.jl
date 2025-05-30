using Printf
using FastHadamardStructuredTransforms_jll

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
    p2SAM = m2SAM = m3SAM = 0.0

    rng = MersenneTwister(seed)
    dist = Uniform(0.0, 1.0)

    tmp1 = zeros(Float64, dim)
    tmp2 = zeros(Float64, dim)

    @inline compute_TMP!(tmp1, tmp2, data(ψ))

    # create arrays of real doubles that will store real and impaginary part of the state across the grays code
    inVR = zeros(Float64, dim)

    @inline p2SAM, m2SAM, m3SAM = sample_MC(L, data(ψ), tmp1, tmp2, inVR)
    m2ADD = copy(m2SAM) # TODO: is this necessary?

    # Create a mask table for the qubits
    MASK_TABLE = UInt[(UInt(1) << (i-1)) for i = 1:L]

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
        @inline tmp1, tmp2 = apply_X_mask_2!(tmp1, tmp2, mask_acc)

        @inline p2SAM, m2SAM, m3SAM = sample_MC(L, data(ψ), tmp1, tmp2, inVR)

        # accept/reject with only one exponentiation
        pβ = m2SAM^β
        if (r < pβ/currPROB) && (currX != 0)
            currPROB = pβ
            currM2 = -log2(m2SAM)
        else
            # undo mask
            currX ⊻= mask_acc

            # CHECK: parallelize applyX is it useful for L large?
            @inline tmp1, tmp2 = apply_X_mask_2!(tmp1, tmp2, mask_acc)
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
            @printf(io, "%d %.20f %.20f %.20f\n", z, res, m2ADD/dim, var)
        end

        # flush every FLUSH_INTERVAL log‐lines:
        # if buf.size ≥ FLUSH_INTERVAL * 60  # rough bytes estimate
        #     write(io, take!(buf)) # write + clear
        # end
    end

    write(io, take!(buf)) # final flush

    close(io)
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

"""
    compute_moments(inVR::AbstractVector{Float64})

Compute the second, fourth, and sixth moments of the input vector inVR.
The moments are computed using SIMD operations for performance.
"""
@fastmath function compute_moments(inVR)
    T = eltype(inVR)
    s2 = zero(T)
    s4 = zero(T)
    s6 = zero(T)

    @inbounds @fastmath @simd for x in inVR
        # x²
        y2 = x*x
        # plain sum of squares
        s2 += y2

        # fuse y2*y2 + s4  →  s4 += x⁴
        s4 = muladd(y2, y2, s4)

        # fuse (x⁴)*x² + s6  →  s6 += x⁶
        s6 = muladd(y2*y2, y2, s6)
    end

    return s2, s4, s6
end

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

@fastmath function sample_MC(
    L::Int,
    X::AbstractVector{ComplexF64},
    tmp1::AbstractVector{Float64},
    tmp2::AbstractVector{Float64},
    inVR::AbstractVector{Float64},
)
    @inline blend_fallback_muladd!(inVR, X, tmp1, tmp2)

    # Perform Fast Hadamard Transform
    call_fht_double!(inVR, Int32(L))

    return @inline PS2, MS2, MS3 = compute_moments(inVR)
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

# const libffht_julia = "/home/jvalle1/git/SRE/src/libffht_julia.so"

# function call_fht_double!(inVR::Vector{Float64}, L::Int32)
#     # Ensure the input array is contiguous in memory
#     inVR_ptr = pointer(inVR)

#     # Perform the ccall
#     ccall(
#         (:fht_double_wrapper, libffht_julia),
#         Cvoid,                            # Return type
#         (Ptr{Cdouble}, Cint),             # Argument types
#         inVR_ptr,
#         L,
#     )

#     return inVR  # Assuming the function modifies the array in place
# end

function call_fht_double!(inVR::Vector{Float64}, L::Int32)
    # Ensure the input array is contiguous in memory
    inVR_ptr = pointer(inVR)

    # Perform the ccall
    ccall(
        (:fht_double, FastHadamardStructuredTransforms_jll.libfasttransforms),
        Cvoid,                            # Return type
        (Ptr{Cdouble}, Cint),             # Argument types
        inVR_ptr,
        L,
    )

    return inVR  # Assuming the function modifies the array in place
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

_compute_chunk_SRE(
    istart::Int64,
    iend::Int64,
    ψ::StateVec{Float64,2},
    Zwhere::Vector{Int64},
    XTAB::Vector{UInt64},
) = _compute_chunk_SRE(0, istart, iend, ψ, Zwhere, XTAB)

@fastmath function _compute_chunk_SRE(
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
        call_fht_double!(inVR, Int32(L))

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
            m2SAM += pnorm ^ 2
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
    site ∉ 1:n && error("site=$(site) out of 0:$(n-1)")

    stride = 3^(site-1) # We subtract one since site is 1-based
    block = 3*stride

    @inbounds for block_start = 1:block:N
        @simd for offset = 0:(stride-1)
            a = block_start + offset
            b = a + stride
            c = b + stride
            # ← this one matches your C++ XX lookup exactly:
            tmp = ψ[a]
            ψ[a] = ψ[b]
            ψ[b] = ψ[c]
            ψ[c] = tmp
        end
    end

    return ψ
end


"""
    fast_hadamard_qutrit!(ψ::AbstractVector{Complex{T}}) where T<:Real

In‐place generalized Hadamard (H₃) on each of the log₃(length(ψ)) qutrits
encoded in the state‐vector `ψ`.  After calling, ψ → (H₃ ⊗ H₃ ⊗ …)·ψ.

Throws if `length(ψ)` isn’t exactly 3ⁿ for some integer n.
"""

# check power of 3
const ωr = -0.5            # = cos(2π/3)
const ωi = √3/2           # = sin(2π/3)

@fastmath function fast_hadamard_qutrit!(
    ψ::AbstractVector{Complex{T}},
) where {T<:AbstractFloat}
    N = length(ψ)
    n = Int(round(log(N)/log(3)))
    N == 3^n || throw(ArgumentError("length(ψ) = $N is not a power of 3"))

    step = 1
    while step < N
        jump = 3*step
        @inbounds for i = 1:jump:N
            @simd for j = 0:(step-1)
                idx = i+j
                a = ψ[idx];
                ar, ai = real(a), imag(a)
                b = ψ[idx+step];
                br, bi = real(b), imag(b)
                c = ψ[idx+2*step];
                cr, ci = real(c), imag(c)

                # compute a + b + c
                r0 = ar + br + cr
                i0 = ai + bi + ci

                # compute a + ω b + ω² c
                #    real: ar + ωr*br − ωi*bi + ωr*cr + ωi*ci
                # imaginary: ai + ωr*bi + ωi*br + ωr*ci − ωi*cr
                r1 = ar + ωr*(br + cr) - ωi*(bi - ci)
                i1 = ai + ωr*(bi + ci) + ωi*(br - cr)

                # compute a + ω² b + ω c
                #    ω² = ωr - i ωi
                r2 = ar + ωr*(br + cr) + ωi*(bi - ci)
                i2 = ai + ωr*(bi + ci) - ωi*(br - cr)

                ψ[idx] = Complex(r0, i0)
                ψ[idx+step] = Complex(r1, i1)
                ψ[idx+2*step] = Complex(r2, i2)
            end
        end
        step = jump
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

    for site = 1:length(XTAB[istart])
        k = XTAB[istart][site] % 3
        if k == 1
            @inline actX_qutrit!(TMP, site)
        elseif k == 2
            # two +1’s is the same as one “–1 mod 3”,
            # but easiest is just call twice
            @inline actX_qutrit!(TMP, site)
            @inline actX_qutrit!(TMP, site)
        end
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
        if ix < dim
            @inline actX_qutrit!(TMP, Zwhere[ix])
        end

    end

    return p2SAM, m2SAM, m3SAM
end
