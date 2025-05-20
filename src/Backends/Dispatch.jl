include("Serial.jl")
include("Threads.jl")

abstract type AbstractBackend end
struct Serial <: AbstractBackend end
struct Threaded <: AbstractBackend end
struct MPIThreads <: AbstractBackend end  # defined even if MPI absent

# ---------- backend selector ----------
function _choose_backend(b::Symbol)
    if b === :serial
        Serial()
    elseif b === :threads
        Threaded()
    elseif b === :mpi_threads
        @assert isdefined(Main, :MPI) "MPI backend requested but MPI.jl is not loaded."
        MPIThreads()
    else
        _auto_backend()
    end
end

function _auto_backend()
    if isdefined(Main, :MPI)
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
        throw(ArgumentError("MPI backend unavailable – `using MPI` first"))
    end

# _apply_backend(::MPIThreads, fsym, args...; kw...) =
#     if isdefined(HadaMAG, :ThreadedBackend)
#         getfield(HadaMAG.ThreadedBackend, fsym)(args...; kw...)
#     else
#         throw(ArgumentError("MPI backend unavailable – `using MPI` first"))
#     end
