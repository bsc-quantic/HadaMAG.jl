# Monte Carlo Stabilizer Rényi Entropy (MC-SRE)
The `HadaMAG.jl` library provides a Monte Carlo method for estimating the Stabilizer Rényi Entropy of order $q$ for a pure quantum state $|ψ⟩$.

The `HadaMAG.MC_SRE(ψ, q; Nsamples=1000, Nβ=13, backend=:auto)` function estimates the SRE using a Monte Carlo sampling approach, where `Nsamples` is the number of samples to draw and `Nβ` is the number of discrete distribution points used in the estimation.

Under the hood, `HadaMAG.MC_SRE(ψ, q)` dispatches to one of several backends (`:serial`, `:threads`, or `:mpi`).
By default, `backend = :auto` chooses the fastest available execution engine. For details on available backends and MPI usage, see the [Backend Configuration](Backends.html) guide.

## Usage
```julia
julia> using HadaMAG

# Prepare a Haar random state of size L with a specified depth
julia> L = 8; ψ = rand_haar(L; depth=2)

# Compute the exact 2nd-order SRE for reference
julia> SRE(ψ, 2)
[==================================================] 100.0%  (256/256)
(6.005977237090554, 6.661338147750939e-16)

# Estimate the 2nd-order SRE using Monte Carlo with 1000 samples
julia> MC_SRE(ψ, 2; Nsamples=1000)
[==================================================] 100.0%  (1000/1000)
5.9989388155066194

# Estimate the 2nd-order SRE using Monte Carlo with 10000 samples
julia> MC_SRE(ψ, 2; Nsamples=10000)
[==================================================] 100.0%  (10000/10000)
6.00198832046363
```