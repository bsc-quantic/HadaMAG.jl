# HadaMAG.jl

![Build Status](https://github.com/bsc-quantic/HadaMAG.jl/actions/workflows/CI.yml/badge.svg?branch=master)

> High-performance Stabilizer Rényi Entropy calculations in Julia

`HadaMAG.jl` is an optimized Julia library for computing the **Stabilizer Rényi Entropy (SRE)** of pure quantum states.
It is designed for both research and high‐performance computing environments, with support for **multi-threading** and **MPI**.

Key features:

- **Exact SRE** – Computes the SRE exactly using the **HadaMAG algorithm**, which applies a sequence of Fast Hadamard Transforms (FHT) to reduce the naive $O(4^n)$ cost down to $O(n 2^n)$.
- **Monte Carlo SRE** – Estimates the SRE using stochastic sampling.
- **Mana for qutrits** – Monte Carlo computation of magic-state mana for qutrit systems.
- **Multiple backends** – Choose between single-threaded, multi-threaded, and MPI+threads execution.
---

## Installation

```julia
julia> using Pkg
julia> Pkg.add("HadaMAG")
```

## Quickstart

```julia
julia> using HadaMAG

# Haar-random 8-qubit state
julia> ψ = rand_haar(8; depth=2)

# Compute the 2nd-order Stabilizer Rényi Entropy
julia> (sre2, lost_norm) = SRE(ψ, 2)
[==================================================] 100.0%  (256/256)
(5.995845930125004, 9.992007221626409e-16)

# Estimate the 2nd-order SRE using Monte Carlo with 10000 samples
julia> sre2_mc = MC_SRE(ψ, 2; Nsamples=10000)
[==================================================] 100.0%  (10000/10000)
5.990668774676943
```

## Backends
HadaMAG supports multiple execution backends:

- `:serial` – single-threaded CPU execution
- `:threads` – multi-threaded CPU execution
- `:mpi` – MPI-based execution for distributed computing

By default, `backend = :auto` selects the fastest available backend.
For more details on configuring and using backends, see the [Backend Configuration guide](https://bsc-quantic.github.io/HadaMAG.jl/dev/manual/Backends/).

## Documentation
Full documentation is available at [HadaMAG.jl Documentation](https://bsc-quantic.github.io/HadaMAG.jl/dev/).

## Acknowledgements

`HadaMAG.jl` uses the [FFHT](https://github.com/FALCONN-LIB/FFHT) C library for efficient bit‐reversed Fast Hadamard Transforms.
