using Test

using HadaMAG

@testset "SRE2" begin
    L = 14
    depth = 4
    ψ = rand_haar(L; depth)

    m2_serial = m2_threads = 0.0

    @testset "SerialBackend" begin

        # test that we get the same results with same seed
        m2_serial = MC_SRE2(ψ; backend = :serial, seed = 123)
        @test MC_SRE2(ψ; backend = :serial, seed = 123) ≈ m2_serial
    end

    @testset "ThreadedBackend" begin

        # test that we get the same results with same seed
        m2_threads = MC_SRE2(ψ; backend = :threads, seed = 123)
        @test MC_SRE2(ψ; backend = :threads, seed = 123) ≈ m2_threads
    end

    # Compare the results from both backends
    @test m2_serial ≈ m2_threads
end