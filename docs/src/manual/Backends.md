# Backend Configuration

HadaMAG can run its kernels on different execution engines depending on your hardware and environment. We call these **backends**.

There are three available backends:

- **`Serial`** – single-threaded CPU execution.
- **`Threaded`** – multi-threaded CPU execution (uses Julia `Threads`).
- **`MPIThreads`** – hybrid MPI + threads execution (one or more threads per MPI rank).

You choose the backend with the `backend` keyword in user-facing functions, e.g.:
```julia
julia> SRE(ψ, q; backend = :auto)   # default is :auto
```
## Backend types and symbols

Internally we define three backend types:

```julia
abstract type AbstractBackend end
struct Serial      <: AbstractBackend end
struct Threaded    <: AbstractBackend end
struct MPIThreads  <: AbstractBackend end  # defined even if MPI is absent
```

These map to the following keyword symbols:
- `backend = :serial` $\to$ `Serial()`.
- `backend = :threads` $\to$ `Threaded()`.
- `backend = :mpi` $\to$ `MPIThreads()`.
- `backend = :auto` (default) $\to$ automatic selection.

You can force an specific backend by passing the corresponding symbol to user functions:
```julia
julia> using HadaMAG

julia> ψ = rand_haar(8; depth=2)

julia> SRE(ψ, 2; backend = :threads)
(6.0095727675204405, 6.661338147750939e-16)
```

## MPI support via `HadaMAGMPIExt` extension
Julia’s package extensions let us ship MPI code without hard-requiring MPI for everyone. Instead, The extension `HadaMAGMPIExt` is automatically discovered by Julia and activated when `MPI.jl` is loaded in your session.

As a user, you need to load `MPI.jl` before using HadaMAG with MPI:
```julia
julia> using HadaMAG

julia> using Pkg; Pkg.add("MPI"); using MPI
Precompiling HadaMAGMPIExt...
  1 dependency successfully precompiled in 2 seconds. 341 already precompiled.
```

The `MPI.jl` package uses `MPIPreferences.jl` to decide which MPI implementation to load (a system MPI or a JLL/bundled MPI).
- Use system MPI (e.g., OpenMPI or MPICH on a cluster):
```julia
julia> using MPIPreferences

julia> MPIPreferences.MPIPreferences.use_system_binary()
```
- Or use a bundled MPI (e.g., `OpenMPI_jll`):
```julia
julia> using MPIPreferences

julia> MPIPreferences.MPIPreferences.use_jll_binary()
```

You can check which MPI you’re using with:
```julia
julia> using MPI

julia> MPI.identify_implementation()
("MPICH", v"4.3.1")
```


This will use the system MPI installation.

### Running using MPI
Here we show a minimal example of running HadaMAG with MPI on a cluster or laptop.
Create a file `run_mpi.jl` with the following content:
```julia
using MPI

using HadaMAG
L = 8
ψ = HadaMAG.rand_haar(L; depth=2)

S, lost = HadaMAG.SRE(ψ, 2; backend = :mpi_threads)

rank = MPI.Comm_rank(MPI.COMM_WORLD)
println("rank=$rank  SRE=$S  lost_norm=$lost")
```

Then run it with `mpiexec` or `mpirun`:
```bash
mpirun -n 4 julia --project yourproject run_sre.jl
```

Or if you are on a cluster with SLURM, you can submit a job script like this:
```bash
srun --ntasks=4 --cpus-per-task=1 julia --project yourproject run_sre.jl
```

#### API Reference
```@docs
HadaMAG.Serial
HadaMAG.Threaded
HadaMAG.MPIThreads
```