# Mana Computation

The `HadaMAG.jl` library also provides functionality to compute the mana of a quantum state for qutrits.
Similar to `HadaMAG.SRE`, the `HadaMAG.Mana(ψ)` function computes the exact mana using Fast Hadamard Transforms (FHT) to reduce the naive ? complexity down to ?.

Under the hood, `HadaMAG.Mana(ψ, q)` dispatches to one of several backends (`:serial`, `:threads`, or `:mpi`).

By default, `backend = :auto` chooses the fastest available execution engine.
For details on available backends and MPI usage, see the [Backend Configuration](Backends.html) guide.

## Usage
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