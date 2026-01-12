# Backend Configuration

HadaMAG can run its kernels on different execution engines depending on your hardware and environment. We call these **backends**.

There are five available backends:

- **`Serial`** – single-threaded CPU execution.
- **`Threaded`** – multi-threaded CPU execution (uses Julia `Threads`).
- **`MPIThreads`** – hybrid MPI + threads execution (requires `MPI.jl`).
- **`CUDA`** – GPU execution using CUDA (requires `CUDA.jl`). Only available for `SRE` function at the moment.
- **`MPI_CUDA`** – hybrid MPI + GPU execution, for multiple nodes with GPUs (requires both `MPI.jl` and `CUDA.jl`). Only available for `SRE` function at the moment.

You choose the backend with the `backend` keyword in user-facing functions, e.g.:
```julia
julia> SRE(ψ, q; backend = :auto) # default is :auto
```
## Backend types and symbols

Internally we define three backend types:

```julia
abstract type AbstractBackend end
struct Serial <: AbstractBackend end
struct Threaded <: AbstractBackend end
struct MPIThreads <: AbstractBackend end
struct CUDAThreads <: AbstractBackend end
struct MPICUDAThreads <: AbstractBackend end
```

These map to the following keyword symbols:
- `backend = :serial` $\to$ `Serial()`.
- `backend = :threads` $\to$ `Threaded()`.
- `backend = :mpi` $\to$ `MPIThreads()`.
- `backend = :cuda` $\to$ `CUDAThreads()`.
- `backend = :mpi_cuda` $\to$ `MPICUDAThreads()`.
- `backend = :auto` (default) $\to$ automatic selection.

You can force an specific backend by passing the corresponding symbol to user functions:
```julia
julia> using HadaMAG

julia> ψ = rand_haar(8; depth=2)
StateVec{ComplexF64,2}(n=8, dim=256, mem=4.04 KiB)

julia> SRE(ψ, 2; backend = :threads)
[==================================================] 100.0%  (256/256)
(3.7603466770760265, 1.3322676295501878e-15)
```

## MPI backend
Julia’s package extensions let us ship MPI code without hard-requiring MPI for everyone. Instead, The extension `HadaMAGMPIExt` is automatically loaded and activated when `MPI.jl` is loaded in your session.

To use it, you just need to add and load `MPI.jl`:
```julia
julia> using HadaMAG

julia> using Pkg; Pkg.add("MPI"); using MPI
Precompiling HadaMAGMPIExt...
  1 dependency successfully precompiled in 2 seconds. 341 already precompiled.
```

### Configuring MPI implementation
The `MPI.jl` package uses `MPIPreferences.jl` to decide which MPI implementation to load (a system MPI or a JLL/bundled MPI).
- Use system MPI (e.g., OpenMPI or MPICH on a cluster):
```julia
julia> using MPIPreferences

julia> MPIPreferences.MPIPreferences.use_system_binary()
```
This will use the system MPI installation.

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

### Running using MPI
Here we show a minimal example of running HadaMAG with MPI on a cluster or laptop.
Create a file `run_mpi.jl` with the following content:
```julia
using HadaMAG
using MPI
using Random

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm)

L = 16
Random.seed!(123) # Fix seed so all ranks generate the same state. You could also generate a state on rank 0 and broadcast it.
ψ = rand_haar(L; depth=5)

m2, lost_norm = SRE(ψ, 2; backend=:mpi_threads, progress=false)

# Only print from rank 0
if rank == 0
    println("SRE(ψ, 2) = ", m2)
    println("Lost norm: ", lost_norm)
end
```

Then run it with `mpiexec` or `mpirun`. For example, to run with 4 processes:
```bash
mpirun -n 4 julia --project yourproject run_sre.jl
```

Or if you are on a cluster with SLURM, you can submit a job script like this:
```bash
srun --ntasks=4 --cpus-per-task=1 julia --project yourproject run_sre.jl
```

## CUDA backend
The CUDA backend is only available for the `SRE` function at the moment. To use it, you just need to add and load `CUDA.jl` into your session:

```julia
julia> using HadaMAG

julia> using Pkg; Pkg.add("CUDA"); using CUDA
```

Then you can call `SRE` with `backend = :cuda`, and you can choose `batch` and `threads` parameters to optimize performance depending on your GPU hardware. Here is a minimal example:
```julia
julia> using HadaMAG
julia> using CUDA

julia> ψ = rand_haar(12; depth=4)
StateVec{ComplexF64,2}(n=12, dim=4096, mem=64.00 KiB)

batch_size = nthreads_per_device = 128 # Adjust these parameters depending on your GPU
128

julia> SRE(ψ, 2; backend = :cuda, progress=false, batch=batch_size, threads=nthreads_per_device)
[==================================================] 100.0%  (4096/4096)
(8.019371115855193, 3.774758283725532e-15)
```

### Multi-GPU with MPI + CUDA
`HadaMAG.jl` also supports hybrid MPI + CUDA execution for multi-GPU systems. To use it, you need to have both `MPI.jl` and `CUDA.jl` loaded in your session.
Then you can call `SRE` with `backend = :mpi_cuda`, and use it similarly to the MPI example above:
```julia
using HadaMAG
using MPI
using Random

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD; rank = MPI.Comm_rank(comm)

L = 16
Random.seed!(123) # Fix seed so all ranks generate the same state. You could also generate a state on rank 0 and broadcast it.
ψ = rand_haar(L; depth=5)

batch_size = nthreads_per_device = 128 # Adjust these parameters depending on your GPU

m2, lost_norm = SRE(ψ, 2; backend=:mpi_cuda, progress=false, batch=batch_size, threads=nthreads_per_device)

# Only print from rank 0
if rank == 0
    println("SRE(ψ, 2) = ", m2)
    println("Lost norm: ", lost_norm)
end
```

#### API Reference
```@docs
HadaMAG.Serial
HadaMAG.Threaded
HadaMAG.MPIThreads
```