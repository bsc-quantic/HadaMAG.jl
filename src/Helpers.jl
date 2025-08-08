using Random
using Distributions
using DelimitedFiles
using Statistics

# Compute whether `len` is an exact power of `q`, returning exponent and a flag.
function _power_q(len::Integer, q::Integer)
    n = 0
    tmp = len
    while tmp % q == 0
        tmp ÷= q
        n += 1
    end
    return n, tmp == 1
end

# This function partitions an integer `n` into `P` parts, returning the counts and displacements.
function partition_counts(n::Integer, P::Integer)
    base, rem = divrem(n, P)
    counts = [i ≤ rem ? base+1 : base for i in 1:P]
    displs = cumsum((0, counts[1:end-1]...))
    return counts, displs
end


# Two Ref containers to hold the function pointer and the default function
const _fht_fn = Ref{Function}()
const _default_fht_fn = Ref{Function}()

# Initialize the default function pointer to the JLL library function
_default_fht_fn[] = (vec, L) -> ccall(
    (:fht_double, FastHadamardStructuredTransforms_jll.libfasttransforms),
    Cvoid, (Ptr{Cdouble}, Cint), pointer(vec), L
)

_fht_fn[] = _default_fht_fn[]


"""
    use_fht_lib(path::String)

Point at your own compiled `.so` that exports exactly the symbol `:fht_double`.
After calling this, every `call_fht!` will invoke your library instead of the JLL one.
"""
function use_fht_lib(path::String)
    handle = dlopen(path, Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND)
    ptr    = dlsym(handle, :fht_double)
    @assert ptr != C_NULL "couldn't find symbol :fht_double in $path"
    @info "Using custom FHT library at $path"
    _fht_fn[] = (vec, L) -> ccall(
        ptr,
        Cvoid, (Ptr{Cdouble}, Cint), pointer(vec), L
    )
    nothing
end

"""
    use_default_fht()

Revert `call_fht!` back to the built-in `FastHadamardStructuredTransforms_jll` implementation.
"""
function use_default_fht()
    @info "Reverting to default FHT library"
    _fht_fn[] = _default_fht_fn[]
    nothing
end

"""
    call_fht!(vec::Vector{Float64}, L::Int32)

In‐place fast Hadamard transform.  After an optional call to `use_fht_lib`,
this will call through your `.so` instead of the default JLL library.
"""
call_fht!(vec::Vector{Float64}, L::Int32) = (_fht_fn[])(vec, L)


"""
    haar_random_unitary(n_qubits::Integer, rng::AbstractRNG = Random.GLOBAL_RNG)

Generate a Haar-distributed random unitary matrix of size 2^n_qubits × 2^n_qubits.

# Arguments
- `n_qubits::Integer`: number of qubits; the output U lives in U(2^n_qubits).
- `rng::AbstractRNG`: random number generator (defaults to `GLOBAL_RNG`).

"""
function haar_random_unitary(n_qubits::Integer, rng::AbstractRNG = Random.GLOBAL_RNG)
    N = 2 ^ n_qubits

    # draw complex Gaussian matrix
    normal_dist = Normal(0.0, 1.0)
    X = reshape(
        [Complex(rand(rng, normal_dist), rand(rng, normal_dist)) for _ = 1:(N^2)],
        N,
        N,
    )

    # thin QR
    F = qr(X)
    Q = Matrix(F.Q)
    R = F.R

    # correct diagonal of R to be positive real
    phases = diag(R) ./ abs.(diag(R))
    return Q * Diagonal(phases)
end

# Process tmp files from the Monte Carlo simulation, extracting means and stds.
function process_files(seed; folder = ".", Nβ = 25)
    # initialize vectors for x-values, means, and stds for second and third rows
    x_vals = Float64[]
    res_means = Float64[]
    res_stds = Float64[]
    m2ADD_means = Float64[]
    m2ADD_stds = Float64[]
    naccepted = Int64[]

    # loop over the range of β values
    for i = 1:Nβ
        # compute β (x value)
        β = Float64(i - 1) / (Nβ - 1) # so that β ∈ [0, 1]
        push!(x_vals, β)

        # construct the filename using string interpolation
        filename = "_$(i)_$(seed+i).dat"
        filename = joinpath(folder, filename)  # prepend folder to filename

        # read the data from file; adjust the delimiter if needed
        isfile(filename) || error("File $filename does not exist.")
        data = readdlm(filename)

        # assume data is a matrix and we need the second and third rows.
        # (If your file has header rows or extra columns, modify the indexing accordingly.)
        row2 = data[:, 2]
        row3 = data[:, 3]

        # compute mean and std for row2 (y values)
        push!(res_means, mean(row2))
        push!(res_stds, std(row2))

        # compute mean and std for row3 (for the second measurement)
        push!(m2ADD_means, mean(row3))
        push!(m2ADD_stds, std(row3))

        # store the number of accepted samples
        push!(naccepted, length(row2))
    end

    return x_vals, res_means, res_stds, m2ADD_means, m2ADD_stds, naccepted
end

# Compute the integral using Simpson's rule.
# This function assumes that the x-values are uniformly spaced.
function integrate_simpson(x::AbstractVector, y::AbstractVector)
    n = length(x) - 1
    # Simpson's rule requires an even number of subintervals (odd number of points)
    if n % 2 != 0
        error(
            "Simpson's rule requires an even number of subintervals (odd number of points).",
        )
    end

    # Compute the spacing assuming uniform x values
    h = (x[end] - x[1]) / n

    # Simpson's rule formula:
    # I = h/3 * [y₀ + yₙ + 4*(y₁ + y₃ + ... + yₙ₋₁) + 2*(y₂ + y₄ + ... + yₙ₋₂)]
    integral = h/3 * (y[1] + y[end] + 4 * sum(y[2:2:(end-1)]) + 2 * sum(y[3:2:(end-2)]))
    return integral
end

"""
    fast_hadamard_qutrit!(ψ::AbstractVector{Complex{T}}) where T<:Real

In‐place generalized Hadamard (H₃) on each of the log₃(length(ψ)) qutrits
encoded in the state‐vector `ψ`.  After calling, ψ → (H₃ ⊗ H₃ ⊗ …)·ψ.

Throws if `length(ψ)` isn’t exactly 3ⁿ for some integer n.
"""

const ωr = -0.5 # = cos(2π/3)
const ωi = √3/2 # = sin(2π/3)

@fastmath function fast_hadamard_qutrit!(
    ψ::AbstractVector{Complex{T}},
) where {T<:AbstractFloat}
    N = length(ψ)
    n = Int(round(log(N)/log(3)))
    # N == 3^n || throw(ArgumentError("length(ψ) = $N is not a power of 3"))

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
