# HadaMAG.jl Documentation

**`HadaMAG.jl`** is an optimized Julia library for computing the Stabilizer Rényi Entropy (SRE) on pure quantum states. Most notably, it contains:

- **Exact SRE**: Computes the exact SRE using the HadaMAG algorithm, which leverages Fast Hadamard Transforms (FHT) to reduce the naive $O(4^n)$ complexity down to $O(n2^n)$ (see [Exact SRE](manual/ExactSRE)).
- **Monte Carlo SRE**: Provides a Monte Carlo method for estimating SRE (see [Monte Carlo SRE](manual/MCSRE)).
- **Mana Computation**: Computes the exact mana of a quantum state for qutrits (see [Mana Computation](manual/Mana)).

!!! warning "Performance tip"
    If you are dealing with significant number of qubits ($N > 16$), you can get **around 30 %** speed-up by compiling and linking your own optimized FFHT library.
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
[==================================================] 100.0%  (256/256)
(6.028326027457565, 1.1102230246251565e-16)

# Estimate the 2nd‐order SRE using Monte Carlo with 10000 samples
julia> sre2_mc = MC_SRE(ψ, 2; Nsamples=10000)
[==================================================] 100.0%  (10000/10000)
6.023172434713934
```

## Manuals
For detailed guides on how to use `HadaMAG.jl`, see the following sections:
- [State Representation](manual/State): How quantum states are represented in `HadaMAG.jl`.
- [Exact SRE](manual/ExactSRE): How to compute the exact Stabilizer Rényi Entropy.
- [Backend Configuration](manual/Backends): How to configure and use different backends for all SRE computation.