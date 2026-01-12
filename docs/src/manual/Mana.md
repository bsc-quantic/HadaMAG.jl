# Mana Computation

The `HadaMAG.jl` library provides functionality to compute the mana of a quantum state for qutrits.
Similar to `HadaMAG.SRE`, the `HadaMAG.Mana(ψ)` function computes the exact mana using Fast Hadamard Transforms (FHT) to reduce the naive ? complexity down to ?.

Under the hood, `HadaMAG.Mana(ψ, q)` dispatches to one of several backends (`:serial`, `:threads`, or `:mpi`).

By default, `backend = :auto` chooses the fastest available execution engine.
For details on available backends and MPI usage, see the [Backend Configuration](Backends.html) guide.

## Basic usage
```julia
julia> using HadaMAG

# Prepare a random state of size L=6
julia> L = 6; amplitudes = rand(ComplexF64, 3^L);

julia> ψ = StateVec(amplitudes; q=3); normalize!(ψ); ψ
StateVec{ComplexF64,3}(n=6, dim=729, mem=11.4 KiB)

# Compute the mana with automatic backend selection (in this case, muli-threaded)
julia> Mana(ψ)
[==================================================] 100.0%  (729/729)
8.556666992528807

# Force the serial backend (not multi-threaded)
julia> Mana(ψ; backend = :serial)
[==================================================] 100.0%  (729/729)
8.556666992528806
```

## Mana for mixed states
The `HadaMAG.jl` library also supports computing the mana of mixed quantum states of qutrit systems, represented as density matrices using the `DensityMatrix{T,q}` struct.
To compute the mana of a mixed state, you can use the `Mana` function, where `ρ` is a `DensityMatrix` object. At the moment, only the serial backend is supported for mana computation of mixed states.

As an example, let's create a mixed state density matrix by partially tracing out a pure state and then compute its mana:
```julia
julia> using HadaMAG

julia> ψ = rand_haar(10; depth=3, q=3) # 10-qutrit Haar-random qutrit state
StateVec{ComplexF64,3}(n=10, dim=59049, mem=923.0 KiB)

julia> ρA = reduced_density_matrix(ψ, 6; side=:right) # Reduced density matrix on 6 qutrits
DensityMatrix{ComplexF64,3}(n=6, size=(729, 729), mem=8.11 MiB)

julia> mana_mixed = Mana(ρA) # Compute the mana of the mixed state
3.7026610515809835
```