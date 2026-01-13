# HadaMAG.jl Documentation

[![arXiv](https://img.shields.io/badge/arXiv-2601.07824-b31b1b.svg)](https://arxiv.org/abs/2601.07824)

**`HadaMAG.jl`** is an optimized Julia library for computing the Stabilizer Rényi Entropy (SRE) on pure quantum states. Most notably, it contains:

- **Exact SRE**: Computes the SRE exactly, which applies a sequence of Fast Hadamard Transforms (FHT) to reduce the naive $O(8^N)$ cost down to $O(N 4^N)$ for $N$ qubits (see [Exact SRE](manual/ExactSRE)).
- **Monte Carlo SRE**: Estimates the SRE using stochastic sampling (see [Monte Carlo SRE](manual/MCSRE)).
- **Mana for qutrit systems**: Computes the exact mana of pure qutrit states, with complexity $O(N 9^N)$ (see [Mana Computation](manual/Mana)).
- **Mana for mixed states of qutrits**: Computes the exact mana for mixed qutrit states represented as density matrices, with complexity $O(N 9^N)$ (see [Mana for Mixed States](manual/Mana/#Mana-for-mixed-states)).

**Paper**: *Computing quantum magic of state vectors*, [arXiv:2601.07824](https://arxiv.org/abs/2601.07824).

!!! warning "Performance tip"
    If you are dealing with significant number of qubits ($N > 16$), you can get **around 30 %** speed-up by compiling and linking your own optimized FFHT library.
    See the [Custom FHT Library](manual/CustomFHT) guide for how to build, enable and revert your own .so.

## Quickstart

```julia
julia> using Pkg; Pkg.add("https://github.com/bsc-quantic/HadaMAG.jl/")

julia> using HadaMAG

# Haar-random 16-qubit state
julia> ψ = rand_haar(16; depth=2)
StateVec{ComplexF64,2}(n=16, dim=65536, mem=1.0 MiB)

# Compute the 2nd-order Stabilizer Rényi Entropy
julia> (sre2, lost_norm) = SRE(ψ, 2)
[==================================================] 100.0%  (65536/65536)
(8.213760134602566, 3.9968028886505635e-15)

# Estimate the 2nd-order SRE using Monte Carlo with 10000 samples
julia> sre2_mc = MC_SRE(ψ, 2; Nsamples=10000)
[==================================================] 100.0%  (10000/10000)
8.218601276704227
```

## Manuals
For detailed guides on how to use `HadaMAG.jl`, see the following sections:
- [State Representation](manual/State): `StateVec` and `DensityMatrix` structs for representing pure and mixed quantum states.
- [Exact SRE](manual/ExactSRE): Stabilizer Rényi Entropy computation via efficient `SRE(ψ, q)` function.
- [Monte Carlo SRE](manual/MCSRE): Estimation of SRE using stochastic sampling with `MC_SRE(ψ, q; Nsamples)`.
- [Mana Computation](manual/Mana): Exact mana computation for pure and mixed qutrit states using `Mana(ψ)` and `Mana(ρ)`.
- [Backend Configuration](manual/Backends): Configuration of different execution backends for optimal performance, including some instructions for MPI and CUDA usage.