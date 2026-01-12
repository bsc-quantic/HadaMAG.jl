include("Serial.jl")
include("Threads.jl")

abstract type AbstractBackend end
struct Serial <: AbstractBackend end
struct Threaded <: AbstractBackend end
struct MPIThreads <: AbstractBackend end  # designed to work when MPI.jl is loaded
struct CUDAThreads <: AbstractBackend end  # designed to work when CUDA.jl is loaded
struct MPICUDAThreads <: AbstractBackend end  # designed to work when both MPI.jl and CUDA.jl are loaded

# ---------- backend selector ----------
function _choose_backend(b::Symbol)
    if b === :serial
        Serial()
    elseif b === :threads
        Threaded()
    elseif b === :mpi_threads
        @assert isdefined(Main, :MPI) "MPI backend requested but MPI.jl is not loaded."
        MPIThreads()
    elseif b === :cuda
        @assert isdefined(Main, :CUDA) "CUDA backend requested but CUDA.jl is not loaded."
        CUDAThreads()
    elseif b === :mpi_cuda
        @assert isdefined(Main, :MPI) "MPI+CUDA backend requested but MPI.jl is not loaded."
        @assert isdefined(Main, :CUDA) "MPI+CUDA backend requested but CUDA.jl is not loaded."
        MPICUDAThreads()
    else
        _auto_backend()
    end
end

function _auto_backend()
    if isdefined(Main, :CUDA) && isdefined(Main, :MPI)
        MPICUDAThreads()
    elseif isdefined(Main, :CUDA)
        CUDAThreads()
    elseif isdefined(Main, :MPI)
        MPIThreads()
    elseif Threads.nthreads() > 1
        Threaded()
    else
        Serial()
    end
end

# ---------- backend call helper ----------
_apply_backend(::Serial, fsym, args...; kw...) =
    getfield(HadaMAG.SerialBackend, fsym)(args...; kw...)

_apply_backend(::Threaded, fsym, args...; kw...) =
    getfield(HadaMAG.ThreadedBackend, fsym)(args...; kw...)

_apply_backend(::MPIThreads, fsym, args...; kw...) =
    if !isnothing(Base.get_extension(HadaMAG, :HadaMAGMPIExt))
        getfield(Base.get_extension(HadaMAG, :HadaMAGMPIExt), fsym)(args...; kw...)
    else
        throw(
            ArgumentError(
                "MPI backend unavailable, ensure MPI.jl is loaded (`using MPI` first)",
            ),
        )
    end

_apply_backend(::CUDAThreads, fsym, args...; kw...) =
    if !isnothing(Base.get_extension(HadaMAG, :HadaMAGCUDAExt))
        getfield(Base.get_extension(HadaMAG, :HadaMAGCUDAExt), fsym)(args...; kw...)
    else
        throw(
            ArgumentError(
                "CUDA backend unavailable, ensure CUDA.jl is loaded (`using CUDA` first)",
            ),
        )
    end

_apply_backend(::MPICUDAThreads, fsym, args...; kw...) =
    if isnothing(Base.get_extension(HadaMAG, :HadaMAGMPIExt))
        throw(
            ArgumentError(
                "MPI+CUDA backend unavailable, ensure MPI.jl is loaded (`using MPI` first)",
            ),
        )
    elseif isnothing(Base.get_extension(HadaMAG, :HadaMAGCUDAExt))
        throw(
            ArgumentError(
                "MPI+CUDA backend unavailable, ensure CUDA.jl is loaded (`using CUDA` first)",
            ),
        )
    else
        getfield(Base.get_extension(HadaMAG, :HadaMAGMPICUDAExt), fsym)(args...; kw...)
    end
