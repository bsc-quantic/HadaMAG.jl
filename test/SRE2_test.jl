@testset "SRE2" begin
    L = 15
    depth = 4
    ψ = rand_haar(L; depth)

    @testset "Monte Carlo SRE2" begin
        m2_serial = m2_threads = 0.0
        Nsamples = 1000
        Nβ = 13

        @testset "SerialBackend" begin
            # test that we get the same results with same seed
            m2_serial = MC_SRE2(ψ; backend = :serial, seed = 123, Nsamples, Nβ)
            @test MC_SRE2(ψ; backend = :serial, seed = 123, Nsamples, Nβ) ≈ m2_serial
        end

        @testset "ThreadedBackend" begin
            # test that we get the same results with same seed
            m2_threads = MC_SRE2(ψ; backend = :threads, seed = 123, Nsamples, Nβ)
            @test MC_SRE2(ψ; backend = :threads, seed = 123, Nsamples, Nβ) ≈ m2_threads
        end

        # Compare the results from both backends
        @test m2_serial ≈ m2_threads
    end

    @testset "Exact SRE2" begin
        m2_exact_serial, _ = SRE2(ψ; backend = :serial)
        m2_exact_threads, _ = SRE2(ψ; backend = :threads)

        # test that we get the same results with both backends
        @test m2_exact_serial ≈ m2_exact_threads
    end

    # Test that Monte Carlo results converge to exact results as Nsamples increases
    # and that the results are improving with more samples
    @testset "Convergence of Monte Carlo" begin
        Nsamples_list = [50, 2000, 50000]
        m2_exact, _ = SRE2(ψ; backend = :threads)

        diff_last = Inf
        for Nsamples in Nsamples_list
            m2_mc = MC_SRE2(ψ; backend = :threads, Nsamples, Nβ = 25)
            @test abs(m2_mc - m2_exact) < 0.1 * abs(m2_exact) # Check if within 10% of exact value

            # Check that the difference is decreasing
            @test diff_last > abs(m2_mc - m2_exact)
            diff_last = abs(m2_mc - m2_exact)
        end
    end

end
