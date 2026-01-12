# HadaMAG.jl

![Build Status](https://github.com/bsc-quantic/HadaMAG.jl/actions/workflows/CI.yml/badge.svg?branch=master)

> High-performance Stabilizer Rényi Entropy calculations in Julia

`HadaMAG.jl` is an optimized Julia library for computing the **Stabilizer Rényi Entropy (SRE)** and **mana** of pure quantum states and **mana** of mixed quantum states.
It is designed for both research and high‐performance computing environments, with support for **multi-threading** and **MPI**.

Key features:

-  **Exact SRE** – Computes the SRE exactly, which applies a sequence of Fast Hadamard Transforms (FHT) to reduce the naive $O(8^N)$ cost down to $O(N 4^N)$ for $N$ qubits.
- **Monte Carlo SRE** – Estimates the SRE using stochastic sampling.
- **Mana for qutrits** – Numerically exact mana calculation for statevectors of qutrit systems, reducing the naive $O(27^N)$ cost down to $O(N 9^N)$ for $N$ qutrits.
- **Mana for mixed states of qutrits** – Numerically exact mana calculation mana for mixed states of qutrits with complexity $O(N 9^N)$.
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

## Backends
`HadaMAG.jl` supports multiple execution backends:

- `:serial` – single-threaded CPU execution.
- `:threads` – multi-threaded CPU execution.
- `:mpi_threads` – MPI-based execution with multi-threading (requires `MPI.jl`).
- `cuda` – GPU execution using CUDA (requires `CUDA.jl`).
- `mpi_cuda` – hybrid MPI + GPU execution for multi-GPU systems (requires both `MPI.jl` and `CUDA.jl`).

By default, `backend = :auto` selects the fastest available backend.
For more details on configuring and using backends, see the [Backend Configuration guide](https://bsc-quantic.github.io/HadaMAG.jl/dev/manual/Backends/).

## Documentation
Full documentation is available at [HadaMAG.jl Documentation](https://bsc-quantic.github.io/HadaMAG.jl/dev/).

## Acknowledgements

`HadaMAG.jl` uses the [FFHT](https://github.com/FALCONN-LIB/FFHT) C library for efficient bit‐reversed Fast Hadamard Transforms.
