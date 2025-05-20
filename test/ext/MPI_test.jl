@testset "MPI" begin
    using MPI
    using Test
    using HadaMAG

    @testset "SRE2" begin
        L = 14
        depth = 4
        ψ = rand_haar(L; depth)

        m2_serial = m2_mpi = 0.0

        @testset "SerialBackend" begin

            # test that we get the same results with same seed
            m2_serial = MC_SRE2(ψ; backend = :serial, seed = 123)
            @test MC_SRE2(ψ; backend = :serial, seed = 123) ≈ m2_serial
        end

        @testset "MPIThreadsBackend" begin

            # test that we get the same results with same seed
            m2_mpi = MC_SRE2(ψ; backend = :mpi_threads, seed = 123)
            @test MC_SRE2(ψ; backend = :mpi_threads, seed = 123) ≈ m2_mpi
        end

        # Compare the results from both backends
        @test m2_serial ≈ m2_mpi
    end

end