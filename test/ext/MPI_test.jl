@testset "MPI" begin
    using MPI

    @testset "SRE2" begin
        L = 14
        depth = 4
        ψ = rand_haar(L; depth)

        @testset "Monte Carlo SRE2" begin
            m2_serial = m2_mpi = 0.0
            Nsamples = 1000
            Nβ = 13

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

        @testset "Exact SRE2" begin
            m2_exact_serial, _ = SRE2(ψ; backend = :serial)
            m2_exact_threads, _ = SRE2(ψ; backend = :mpi_threads)

            # test that we get the same results with both backends
            @test m2_exact_serial ≈ m2_exact_threads
        end
    end
end