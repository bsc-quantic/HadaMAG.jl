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

        # generate a random number of qubits to flip
        whereA_vec = rand(rng, 1:L, tr)

        # obtain the mask for the qubits to flip
        mask_acc = foldl(⊻, MASK_TABLE[whereA_vec]; init = zero(UInt))

        # apply mask
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
            @inline tmp1, tmp2 = apply_X_mask_2!(tmp1, tmp2, mask_acc)
        end

        # update stats
        sum_p += currM2
        sum_p2 += currM2^2
        n_p += 1

        # occasionally dump to disk
        ile = z > 1 ? 2^(Int(trunc(log2(z)/1.5))) : 10
        if (z % ile == 0) && (n_p > 20)
            res = sum_p / n_p
            var = sum_p2 / n_p
            @printf(buf, "%d %.20f %.20f %.20f\n", z, res, m2ADD/dim, var)
        end

        # flush every FLUSH_INTERVAL log‐lines:
        if buf.size ≥ FLUSH_INTERVAL * 60  # rough bytes estimate
            write(io, take!(buf)) # write + clear
        end
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
