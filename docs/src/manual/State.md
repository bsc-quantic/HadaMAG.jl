# State representation in `HadaMAG.jl`

In **HadaMAG.jl**, the `StateVec{T,q}` type provides a representation of an $n$-qudit quantum state in the computational basis (with local dimension $q$), and offers support for constructing from raw amplitude vectors, generating Haar-random states, loading from common on-disk formats and other utilities.

## Constructing a `StateVec{T,q}`
Create a state from an existing amplitude vector of length = $q^n$:
```julia
julia> using HadaMAG

julia> amplitudes = randn(ComplexF64, 2^4);

julia> ψ = StateVec(amplitudes) # defaults to q=2 (qubits)
StateVec{ComplexF64,2}(n=4, dim=16, mem=296.0 B)
```

Or specify a non-qubit local dimension `q` (e.g., for qutrits, `q=3`):
```julia
julia> amplitudes = randn(ComplexF64, 3^3);

julia> ψ = StateVec(amplitudes; q=3) # qutrits
StateVec{ComplexF64,3}(n=3, dim=27, mem=472.0 B)
```

## Density Matrices
**HadaMAG.jl** also provides with the `DensityMatrix{T,q}` struct to represent mixed states. Additionally, you can compute the reduced density matrix of a pure state `ψ::StateVec{T,q}` using the `reduced_density_matrix` function:
```julia
julia> ψ = StateVec(amplitudes)
StateVec{ComplexF64,2}(n=4, dim=16, mem=296.0 B)

julia> ρA = reduced_density_matrix(ψ, 2; side=:right)
DensityMatrix{ComplexF64,2}(n=2, size=(4, 4), mem=304.0 B)
```

### Generating quantum circuit states
`HadaMAG.jl` provides the `rand_haar` function to generate pure states obtained with quantum circuits from computational basis state, by applying Haar random 2-qudit gates in a brickwall pattern with an specified `depth`.
For example, let's generate a state on `N=4` qubits, corresponding to `depth=3` layers of random 2-qubit gates:

```julia
julia> using HadaMAG

julia> N = 4
4

julia> depth = 3
3

julia> ψ = rand_haar(N; depth)
StateVec{ComplexF64,2}(n=4, dim=16, mem=296.0 B)
```

You can also generate states evolved with quantum circuits for qutrits:

```julia
julia> ψ_qutrit = rand_haar(3; q=3, depth=2)
StateVec{ComplexF64,3}(n=3, dim=27, mem=472.0 B)
```

## Loading a state vector from disk
You can load amplitude data from various on-disk formats into a `StateVec`. The data is **not renormalized** automatically, so make sure your file encodes a unit-norm state.

Supported formats (by extension):
- `.jld2`: JLD2 format (with `JLD2.jl`)
- `.npy`: NumPy format (with `NPZ.jl`)
- `.txt` and other text formats: Whitespace-delimited real and imaginary parts.

```julia
julia> using LinearAlgebra

julia> ψ = load_state("state_L_12.txt")
StateVec{ComplexF64,2}(n=12, dim=4096, mem=64.0 KiB)

julia> norm(ψ)
1.0
```

### API Reference

```@docs
HadaMAG.StateVec
HadaMAG.rand_haar
HadaMAG.load_state
HadaMAG.apply_gate!
```
