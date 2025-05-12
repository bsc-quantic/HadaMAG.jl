using Random
using Distributions

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
