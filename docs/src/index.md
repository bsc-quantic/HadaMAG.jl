# HadaMAG.jl Documentation

**`HadaMAG.jl`** is an optimized Julia library for computing Stabilizer Rényi Entropy (SRE) on pure quantum states. Most notably, it contains:

- **Exact SRE**: Computes the exact SRE using the HadaMAG algorithm, which leverages Fast Hadamard Transforms (FHT) to reduce the naive $O(4^n)$ complexity to $O(n2^n)$.
- **Monte Carlo SRE**: Provides a Monte Carlo method for estimating SRE.
- **Mana Computation**: Computes the mana of a quantum state for qutrits in a Monte Carlo fashion.

!!! warning "Performance tip"
    If you’re processing large vectors or doing millions of transforms, you can get **up to 30 %** more speed by compiling and linking your own optimized FFHT library.
    See the [Custom FHT Library](manual/CustomFHT) guide for how to build, enable and revert your own .so.

## Quickstart

```julia
julia> using Pkg; Pkg.add("HadaMAG")

julia> using HadaMAG

# Prepare a Haar-random 8-qubit state
julia> ψ = rand_haar(8; depth=2)
StateVec{ComplexF64,2}(n=8, dim=256, mem=4.04 KiB)

# Compute the 2nd‐order Stabilizer Rényi Entropy
julia> (sre2, lost_norm) = SRE(ψ, 2)
(6.028326027457565, 1.1102230246251565e-16)

# Estimate the 2nd‐order SRE using Monte Carlo with 1000 samples
julia> sre2_mc = MC_SRE(ψ, 2; Nsamples=10000)
6.023172434713934
```

## Manuals
For detailed guides on how to use `HadaMAG.jl`, see the following sections:
- [State Representation](manual/State): How quantum states are represented in `HadaMAG.jl`.
- [Exact SRE](manual/ExactSRE): How to compute the exact Stabilizer Rényi Entropy.
- [Monte Carlo SRE](manual/MC_SRE): How to estimate the Stabilizer Rényi Entropy using Monte Carlo methods.
- [Mana Computation](manual/Mana): How to compute the mana of a quantum state