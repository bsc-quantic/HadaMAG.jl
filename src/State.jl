using Random
using LinearAlgebra
using JLD2
using NPZ

"""
    StateVec

A lightweight container for a pure quantum state of `n` *q*-dits
(`q` = qudit dimension, 2 for qubits) stored in the computational basis
|0⋯00⟩, |0⋯01⟩, …, |q−1⋯(q−1)⟩.
"""
struct StateVec{T,q}
    data :: Vector{Complex{T}}
    n    :: Int
    q    :: Int
end

"""
    StateVec(vec::AbstractVector{<:Complex}; q::Int = 2)

Create a [`StateVec`](@ref) from an existing amplitude vector `vec`.
Throws `ArgumentError` if `length(vec)` is not an exact power of `q`.

# Arguments

- `vec`: Vector of complex amplitudes.
- `q`: Dimension of each qudit (default = 2 for qubits).

# Returns

- A [`StateVec`](@ref) containing a copy of `vec` and inferred `n` & `q`.

# Example

```julia-repl
julia> ψ = randn(ComplexF64, 2^4);
julia> StateVec(ψ) # defaults to qubits
StateVec{Float64,2}(n=4, q=2)
```
"""
function StateVec(vec::AbstractVector{<:Complex{T}}; q::Int = 2) where T
    n, ispow = _power_q(length(vec), q)
    ispow || throw(ArgumentError("length(vec)=$(length(vec)) is not a power of q=$q"))
    return StateVec{T,q}(Vector{Complex{T}}(vec), n, q)
end

Base.size(s::StateVec) = (length(s.data),)
Base.getindex(s::StateVec, i::Int) = s.data[i]

# Pretty-print summary in the REPL
function Base.show(io::IO, ::MIME"text/plain", s::StateVec)
    print(io, "StateVec{", eltype(s.data), ",", s.q, "}(n=", s.n, ", dim=", length(s.data), ")")
end

"""
    rand_haar(n::Int; q::Int = 2, T = Float64, rng = Random.GLOBAL_RNG)

Generate a Haar-random pure state on `n` qudits of dimension `q`, normalized to unit norm.

# Arguments

- `n`: Number of qudits.
- `q`: Qudit dimension (default = 2).
- `T`: Floating-point type (default = Float64).
- `rng`: Random number generator (default = `Random.GLOBAL_RNG`).

# Returns

- A [`StateVec`](@ref) with random amplitudes.
"""
function rand_haar(n::Int; q::Int = 2, T = Float64, rng::AbstractRNG = Random.GLOBAL_RNG)
    vec = randn(rng, Complex{T}, q^n)
    vec ./= norm(vec)
    return StateVec(vec; q=q)
end

"""
    load_state(path::AbstractString; q::Int = 2) -> StateVec

Load a state vector from disk into a [`StateVec`](@ref).
The loaded vector is not renormalized; ensure it has unit norm if required.

Supported formats (by file extension):

- `.jld2`: Reads dataset `"state"` via JLD2.jl.
- `.npy`:  Reads NumPy array via NPZ.jl.
- Otherwise: Whitespace-delimited real and imaginary parts.

# Arguments

- `path`: Path to the file containing state amplitudes.
- `q`: Dimension of each qudit (default = 2).

# Returns

- A [`StateVec`](@ref) constructed from the loaded data.

"""
function load_state(path::AbstractString; q::Int = 2)
    if endswith(path, ".jld2")
        data = JLD2.load(path, "state")
        return StateVec(data; q=q)
    elseif endswith(path, ".npy")
        data = NPZ.npzread(path)
        return StateVec(data; q=q)
    else
        raw = read(path, String)
        vals = parse.(Float64, split(raw))
        length(vals) % 2 == 0 || throw(ArgumentError("File does not contain real/imaginary pairs"))
        real_parts = vals[1:2:end]
        imag_parts = vals[2:2:end]
        return StateVec(complex.(real_parts, imag_parts); q=q)
    end
end
