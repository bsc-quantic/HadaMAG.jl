include("Serial.jl")
include("Threads.jl")

abstract type AbstractBackend end
struct Serial <: AbstractBackend end
struct Threads <: AbstractBackend end
struct MPIThreads <: AbstractBackend end  # defined even if MPI absent

# ---------- backend selector ----------
function _choose_backend(b::Symbol)
    if b === :serial
        Serial()
    elseif b === :threads
        Threads()
    elseif b === :mpi_threads
        @assert isdefined(Main, :MPI) "MPI backend requested but MPI.jl is not loaded."
        MPIThreads()
    else
        _auto_backend()
    end
end

function _auto_backend()
    if isdefined(Main, :MPI) && MPI.Initialized()
        MPIThreads()
    elseif Threads.nthreads() > 1
        Threads()
    else
        Serial()
    end
end

# ---------- backend call helper ----------
_apply_backend(::Serial, fsym, args...; kw...) =
    getfield(HadaMAG.SerialBackend, fsym)(args...; kw...)

_apply_backend(::Threads, fsym, args...; kw...) =
    getfield(HadaMAG.ThreadsBackend, fsym)(args...; kw...)

_apply_backend(::MPIThreads, fsym, args...; kw...) =
    if isdefined(HadaMAG, :MPIThreadsBackend)
        getfield(HadaMAG.MPIThreadsBackend, fsym)(args...; kw...)
    else
        throw(ArgumentError("MPI backend unavailable – `using MPI` first"))
    end
