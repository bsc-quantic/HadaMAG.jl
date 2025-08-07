# State representation in `HadaMAG.jl`

In **HadaMAG.jl**, the `StateVec{T,q}` type provides a simple, efficient representation of an $n$-qudit (local dimension $q$) quantum state in the computational basis, with support for constructing from raw amplitude vectors, generating Haar-random states, loading from common on-disk formats and other utilities.

## Constructing a `StateVec{T,q}`
Create from an existing amplitude vector (must have length = $q^n$):

```julia
julia> using HadaMAG

julia> amplitudes = randn(ComplexF64, 2^4);

julia> ψ = StateVec(amplitudes) # defaults to q=2 (qubits)
StateVec{ComplexF64,2}(n=4, dim=16, mem=296.0 B)
```

Or specify a non-qubit local dimension:
```julia
julia> amplitudes = randn(ComplexF64, 3^3);

julia> ψ = StateVec(amplitudes; q=3) # qutrits
StateVec{ComplexF64,3}(n=3, dim=27, mem=472.0 B)
```

### Sampling Haar-random states
Generate a Haar-random state on `n` qubits, using `depth` layers of random 2-qubit gates:

```julia
julia> ψ = rand_haar(4; depth=3)
StateVec{Float64,2}(n=4, dim=16, mem=...)
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