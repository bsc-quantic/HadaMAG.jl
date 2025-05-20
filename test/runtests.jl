using HadaMAG
using Test

@testset "Unit tests" begin
    include("State_test.jl")
    include("SRE2_test.jl")
end

@testset "Integration tests" begin
    include("ext/MPI_test.jl")
end