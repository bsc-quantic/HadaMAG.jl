# HadaMAG.jl

![Build Status](https://github.com/bsc-quantic/HadaMAG.jl/actions/workflows/CI.yml/badge.svg?branch=master)  [![arXiv](https://img.shields.io/badge/arXiv-2601.07824-b31b1b.svg)](https://arxiv.org/abs/2601.07824)

> High-performance Stabilizer Rényi Entropy and Mana computations in Julia

`HadaMAG.jl` is an optimized Julia library for computing the **Stabilizer Rényi Entropy (SRE)** and **Mana** of pure quantum states, and **Mana** of mixed quantum states.
It is designed for both research and high‐performance computing environments, with support for **multi-threading**, **MPI**, **GPU acceleration** (using CUDA), and **multi-GPU** systems (using MPI + CUDA).

Paper: [https://arxiv.org/abs/2601.07824](https://arxiv.org/abs/2601.07824)

Key features:
-  **Exact SRE** – Computes the SRE exactly, which applies a sequence of Fast Hadamard Transforms (FHT) to reduce the naive $O(8^N)$ cost down to $O(N 4^N)$ for $N$ qubits.
- **Monte Carlo SRE** – Estimates the SRE using stochastic sampling.
- **Mana for qutrits** – Numerically exact mana computation for statevectors of qutrit systems, reducing the naive $O(27^N)$ cost down to $O(N 9^N)$ for $N$ qutrits.
- **Mana for mixed states of qutrits** – Numerically exact mana computation mana for mixed states of qutrits, with complexity $O(N 9^N)$.
- **Multiple backends** – Choose between different execution backends (`:serial`, `:threads`, `:mpi_threads`, `:cuda`, `:mpi_cuda`) for optimal performance on your hardware.
---



## Installation
To install `HadaMAG.jl`, use the Julia package manager:
```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/bsc-quantic/HadaMAG.jl/")
```

## Stabilizer Rényi Entropy (SRE)
The Stabilizer Rényi Entropy (SRE) of order $q$ is a measure of the non-stabilizerness of a pure quantum state $| ψ ⟩$.
`HadaMAG.jl` provides both exact and Monte Carlo methods to compute the SRE efficiently:
```julia
julia> using HadaMAG

# Haar-random 16-qubit state
julia> ψ = rand_haar(16; depth=2)
StateVec{ComplexF64,2}(n=16, dim=65536, mem=1.0 MiB)

# Compute the q=2 Stabilizer Rényi Entropy
julia> (sre2, lost_norm) = SRE(ψ, 2)
[==================================================] 100.0%  (65536/65536)
(8.213760134602566, 3.9968028886505635e-15)

# Estimate the q=2 SRE using Monte Carlo with 10000 samples
julia> sre2_mc = MC_SRE(ψ, 2; Nsamples=10000)
[==================================================] 100.0%  (10000/10000)
8.218601276704227
```

## Mana Computation
`HadaMAG.jl` provides functionality to compute the mana of pure states for qutrit systems:
```julia
julia> using HadaMAG

# Haar-random 10-qutrit state
julia> ψ = rand_haar(10; depth=3, q=3)
StateVec{ComplexF64,3}(n=10, dim=59049, mem=923.0 KiB)

# Compute the mana
julia> mana = Mana(ψ)
4.720097873481441
```

Additionally, you can compute the mana of mixed quantum states of qutrit systems, represented as density matrices using the `DensityMatrix{T,q}` struct:
```julia
julia> using HadaMAG

# Haar-random 10-qutrit state
julia> ψ = rand_haar(10; depth=3, q=3)
StateVec{ComplexF64,3}(n=10, dim=59049, mem=923.0 KiB)

# Reduced density matrix by tracing out part of the system
julia> ρA = reduced_density_matrix(ψ, 6)
DensityMatrix{ComplexF64,3}(n=6, size=(729, 729), mem=8.11 MiB)

# Compute the mana of the mixed state
julia> mana_mixed = Mana(ρA)
3.5551656874727082
```

### Backends
`HadaMAG.jl` supports multiple execution backends:

- `:serial` – single-threaded CPU execution.
- `:threads` – multi-threaded CPU execution.
- `:mpi_threads` – MPI-based execution with multi-threading (requires `MPI.jl`).
- `:cuda` – GPU execution using CUDA (requires `CUDA.jl`).
- `:mpi_cuda` – hybrid MPI + GPU execution for multi-GPU systems (requires both `MPI.jl` and `CUDA.jl`).

By default, `backend = :auto` selects the fastest available backend.
For more details on configuring and using backends, see the [Backend Configuration guide](https://bsc-quantic.github.io/HadaMAG.jl/dev/manual/Backends/).

### Documentation
Full documentation is available at [HadaMAG.jl Documentation](https://bsc-quantic.github.io/HadaMAG.jl/dev/).

### Acknowledgements

`HadaMAG.jl` uses the [FFHT](https://github.com/FALCONN-LIB/FFHT) C library for efficient bit‐reversed Fast Hadamard Transforms.
