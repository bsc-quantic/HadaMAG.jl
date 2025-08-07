# Exact Stabilizer Rényi Entropy (SRE)

The Stabilizer Rényi Entropy (SRE) of order $q$ is a measure of the non-stabilizer content of a pure quantum state $| ψ ⟩$.
Under the hood, `HadaMAG.SRE(ψ, q)` dispatches to one of several backends (`:serial`, `:threads`, or `:mpi`) and uses a sequence of Fast Hadamard Transforms to reduce the naive $O(4^n)$ sum down to $O(n2^n)$ work.

By default, `backend = :auto` picks the best available engine, but you can override it if you want to force single-threaded, multi-threaded, or MPI-based execution.
For more on choosing backends, see the [Backend Configuration](backends.html) guide.

## Usage

```julia
julia> using HadaMAG

# Prepare a Haar random state of size L with a specified depth
julia> L = 8; ψ = rand_haar(L; depth=2)

# Compute the 2nd-order SRE with automatic backend selection (in this case, muli-threaded)
julia> SRE(ψ, 2)
(6.009572767520443, 4.440892098500626e-16)

# Force the serial backend (not multi-threaded)
julia> SRE(ψ, 2; backend = :serial)
(6.0095727675204405, 6.661338147750939e-16)
```

In each call you get a tuple (entropy, lost_norm), where lost_norm is the numerical deviation from unit norm after the sequence of FHTs.

### API Reference

```@docs
HadaMAG.SRE
```