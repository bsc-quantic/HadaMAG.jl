using Test

@testset "MPI" begin
    using MPI
    using HadaMAG
    using Random

    @testset "SRE2" begin
        L = 14
        depth = 4

        Random.seed!(1234)
        ψ = rand_haar(L; depth)

        @testset "Monte Carlo SRE2" begin
            m2_serial = m2_mpi = 0.0
            Nsamples = 1000
            Nβ = 13

            @testset "SerialBackend" begin

                # test that we get the same results with same seed
                m2_serial = MC_SRE(ψ, 2; backend = :serial, seed = 123, progress = false)
                @test MC_SRE(ψ, 2; backend = :serial, seed = 123, progress = false) ≈
                      m2_serial
            end

            @testset "MPIThreadsBackend" begin

                # test that we get the same results with same seed
                m2_mpi = MC_SRE(ψ, 2; backend = :mpi_threads, seed = 123, progress = false)
                @test MC_SRE(ψ, 2; backend = :mpi_threads, seed = 123, progress = false) ≈
                      m2_mpi
            end

            # Compare the results from both backends
            @test m2_serial ≈ m2_mpi
        end

        @testset "Exact SRE2" begin
            m2_exact_serial, lost_norm_serial =
                SRE(ψ, 2; backend = :serial, progress = false)
            m2_exact_threads, lost_norm_threads =
                SRE(ψ, 2; backend = :mpi_threads, progress = false)

            # test that we get the same results with both backends
            @test m2_exact_serial ≈ m2_exact_threads
        end
    end
end
