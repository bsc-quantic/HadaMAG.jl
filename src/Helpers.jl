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

    # loop over j = 0 to 12 (inclusive)
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
