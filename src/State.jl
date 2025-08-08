using Random
using LinearAlgebra
using JLD2
using NPZ

"""# HadaMAG.jl: StateVec
A lightweight container for a pure quantum state of `n` *q*-dits
(`q` = qudit dimension, 2 for qubits) stored in the computational basis
|0⋯00⟩, |0⋯01⟩, …, |q−1⋯(q−1)⟩.
"""
struct StateVec{T,q}
    data::Vector{Complex{T}}
    n::Int
    q::Int
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
function StateVec(vec::AbstractVector{<:Complex{T}}; q::Int = 2) where {T}
    n, ispow = _power_q(length(vec), q)
    ispow || throw(ArgumentError("length(vec)=$(length(vec)) is not a power of q=$q"))
    return StateVec{T,q}(Vector{Complex{T}}(vec), n, q)
end

qudits(s::StateVec{T,q}) where {T,q} = s.n
qubits(s::StateVec{T,2}) where {T} = s.n
qudit_dim(s::StateVec{T,q}) where {T,q} = s.q
data(s::StateVec) = s.data

Base.size(s::StateVec) = (length(s.data),)
Base.getindex(s::StateVec, i::Int) = s.data[i]
Base.copy(s::StateVec) = StateVec(s.data; q = s.q)
Base.isequal(s1::StateVec, s2::StateVec) = isequal(s1.data, s2.data) && s1.q == s2.q
Base.isapprox(s1::StateVec, s2::StateVec; atol = 1e-8) =
    isapprox(s1.data, s2.data; atol = atol) && s1.q == s2.q

LinearAlgebra.norm(s::StateVec) = norm(s.data)
LinearAlgebra.normalize!(s::StateVec) = normalize!(s.data)
LinearAlgebra.normalize(s::StateVec) = normalize(s.data)

# Pretty-print summary in the REPL
function Base.show(io::IO, ::MIME"text/plain", s::StateVec)
    # total bytes of the buffer (including any GC-allocated overhead)
    bytes = Base.summarysize(s.data)
    # human-friendly units (optional)
    function _fmt(n)
        for u in ("B", "KiB", "MiB", "GiB")
            if n < 1024
                return "$(round(n, sigdigits=3)) $u"
            end
            n /= 1024
        end
        return "$(round(n, sigdigits=3)) TiB"
    end
    memstr = _fmt(bytes)

    print(
        io,
        "StateVec{",
        eltype(s.data),
        ",",
        s.q,
        "}",
        "(n=",
        s.n,
        ", dim=",
        length(s.data),
        ", mem=",
        memstr,
        ")",
    )
end

"""
    rand_haar(n::Int, depth::Int; rng::AbstractRNG = Random.GLOBAL_RNG)

Generate a Haar-random state on `n` qubits of local dimension `2`, normalized to unit norm.

# Arguments
- `depth::Int`: number of layers of random 2-qudit gates (brick-wall pattern) to apply to a state vector
  initialized with iid complex Gaussian entries.

# Keyword Arguments
- `rng::AbstractRNG`: random number generator (default = `Random.GLOBAL_RNG`).
"""
function rand_haar(n::Int; depth, rng::AbstractRNG = Random.GLOBAL_RNG)
    dim = 2^n
    dist = Normal(0.0, 1.0)
    vec = [Complex(rand(rng, dist), rand(rng, dist)) for i = 1:dim]

    normalize!(vec)
    apply_brick_wall_haar!(vec, n, depth, rng)

    return StateVec(vec; q = 2)
end

function rand_cliff(n::Int; depth::Int = 1, rng::AbstractRNG = Random.GLOBAL_RNG)
    """
    Generate a random Clifford circuit on `n` qubits with `depth` layers.
    Each layer consists of Haar-random 2-qubit gates applied in a brick-wall pattern.
    """
    state = zeros(ComplexF64, 2^n)
    state[1] = 1.0  # Initialize to |0...0⟩ state

    apply_brick_wall_cliff!(state, n, depth, rng)

    return StateVec(state; q = 2)
end

function apply_brick_wall_cliff!(
    state::AbstractVector{ComplexF64},
    nqubits::Integer,
    depth::Integer,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    """
    Apply a brick-wall pattern of Clifford gates to `state`.
    The circuit has `depth` layers; odd layers act on qubits (1,2),(3,4)…, even on (2,3),(4,5)…
    """
    @assert length(state) == (1 << nqubits) "apply_brick_wall_cliff only supports qubit state vectors of length 2^n"

    # 1qubit gates
    one_gates = [
        [1.0 0.0; 0.0 1.0],  # I
        [0.0 1.0; 1.0 0.0],  # X
        [0.0 -im; im 0.0],   # Y
        [1.0 0.0; 0.0 -1.0], # Z
        [1.0 0.0; 0.0 1.0],  # H
        [1.0 0.0; 0.0 1*im], # S
    ]

    cnot = [
        1.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0;
        0.0 0.0 0.0 1.0;
        0.0 0.0 1.0 0.0; # CNOT
    ]

    # 2qubit gates (cnot)
    two_gates = [cnot]

    gates = vcat(one_gates, two_gates)

    for layer = 1:depth
        # choose starting qubit: 1→(1,2),(3,4)… ; 2→(2,3),(4,5)…
        start = isodd(layer) ? 1 : 2
        for q = start:2:(nqubits-1)
            gate = rand(rng, gates)
            if size(gate) == (2, 2)
                # convert to 4x4 gate for 2-qubit application
                gate = kron(gate, [1.0 0.0; 0.0 1.0]) # tensor product with identity
            end
            apply_2gate!(state, gate, q, q+1)
        end
    end
    return state
end

"""
    apply_brick_wall_haar!(state::AbstractVector{ComplexF64}, nqubits::Integer, depth::Integer; rng::AbstractRNG = Random.GLOBAL_RNG)

Apply an in-place “brick-wall” of Haar-random 2-qubit unitaries to `state`.
The circuit has `depth` layers; odd layers act on qubits (1,2),(3,4)…, even on (2,3),(4,5)…

# Arguments
- `state`: length-2^nqubits state vector (will be mutated).
- `nqubits`: number of qubits.
- `depth`: number of alternating layers.
- `rng`: keyword RNG (defaults to `GLOBAL_RNG`).
"""
@inline function apply_brick_wall_haar!(
    state::AbstractVector{ComplexF64},
    nqubits::Integer,
    depth::Integer,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    @assert length(state) == (1 << nqubits) "apply_brick_wall_haar! only supports qubit state vectors of length 2^n"
    for layer = 1:depth
        # choose starting qubit: 1→(1,2),(3,4)… ; 2→(2,3),(4,5)…
        start = isodd(layer) ? 1 : 2
        for q = start:2:(nqubits-1)
            U = haar_random_unitary(2, rng)
            apply_2gate!(state, U, q, q+1)
        end
    end
    return state
end

"""
    apply_brick_wall_haar!(ψ::StateVec{T,2}, L::Int, depth::Int, rng::AbstractRNG = Random.GLOBAL_RNG)

Apply a brick-wall Haar-random circuit to a state vector `vec` in-place.
The circuit consists of `depth` layers of random unitary 2-qubit gates, alternating between even and odd qubit pairs.

# Arguments
- `ψ`: [`StateVec`](@ref) to be modified.
- `depth`: Number of layers of gates.
- `rng`: Random number generator (default = `Random.GLOBAL_RNG`).
"""
apply_brick_wall_haar!(
    ψ::StateVec{T,2},
    depth::Int,
    rng::AbstractRNG = Random.GLOBAL_RNG,
) where {T} = apply_brick_wall_haar!(ψ.data, ψ.n, depth, rng)

"""
    apply_2gate!(state::AbstractVector{Complex{T}}, gate::AbstractMatrix{Complex{T}}, q1::Int, q2::Int)

Apply in-place a two-qubit `gate` (4×4 matrix) to qubits `q1`,`q2` (1-based) on `state`.
`state` must have length `2^n` and contain amplitude data in computational basis.
"""
function apply_2gate!(
    state::AbstractVector{Complex{T}},
    gate::AbstractMatrix{Complex{T}},
    q1::Int,
    q2::Int,
) where {T}
    N = length(state)

    @assert N & (N-1) == 0 "apply_2gate! only supports qubit state vectors of length 2^n"
    @assert size(gate) == (4, 4) "apply_2gate! only supports 4x4 gates"

    tmp = zeros(Complex{T}, N)
    # precompute bitmasks
    mask1 = 1 << (q1-1)
    mask2 = 1 << (q2-1)
    mask12 = mask1 | mask2
    for i = 0:(N-1)
        @inbounds begin
            amp = state[i+1]
            # extract qubit bits
            b1 = (i & mask1) >>> (q1-1)
            b2 = (i & mask2) >>> (q2-1)
            idx_in = (b1 << 1) | b2
            base = i & ~mask12
            # distribute amplitudes
            for o = 0:3
                j1 = (o >>> 1) & 1
                j2 = o & 1
                j = base | (j1 << (q1-1)) | (j2 << (q2-1))
                tmp[j+1] += gate[o+1, idx_in+1] * amp
            end
        end
    end
    state .= tmp
    return state
end

"""
    apply_2gate!(sv::StateVec{T,2}, gate::AbstractMatrix{Complex{T}}, q1::Int, q2::Int)

Apply a two-qubit gate directly on a [`StateVec{T,2}`](@ref).
"""
function apply_2gate!(
    sv::StateVec{T,2},
    gate::AbstractMatrix{Complex{T}},
    q1::Int,
    q2::Int,
) where {T}
    apply_2gate!(sv.data, gate, q1, q2)
    return sv
end

apply_2gate(
    sv::StateVec{T,2},
    gate::AbstractMatrix{Complex{T}},
    q1::Int,
    q2::Int,
) where {T} = apply_2gate!(copy(sv), gate, q1, q2)

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
        return StateVec(data; q = q)
    elseif endswith(path, ".npy")
        data = NPZ.npzread(path)
        return StateVec(data; q = q)
    else
        raw = read(path, String)
        vals = parse.(Float64, split(raw))
        length(vals) % 2 == 0 ||
            throw(ArgumentError("File does not contain real/imaginary pairs"))
        real_parts = vals[1:2:end]
        imag_parts = vals[2:2:end]
        return StateVec(complex.(real_parts, imag_parts); q = q)
    end
end
