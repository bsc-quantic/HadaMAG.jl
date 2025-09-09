using Random
using Statistics

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
            m2_serial =
                MC_SRE(ψ, 2; backend = :serial, seed = 123, Nsamples, Nβ, progress = false)
            @test MC_SRE(
                ψ,
                2;
                backend = :serial,
                seed = 123,
                Nsamples,
                Nβ,
                progress = false,
            ) ≈ m2_serial
        end

        @testset "ThreadedBackend" begin
            # test that we get the same results with same seed
            m2_threads =
                MC_SRE(ψ, 2; backend = :threads, seed = 123, Nsamples, Nβ, progress = false)
            @test MC_SRE(
                ψ,
                2;
                backend = :threads,
                seed = 123,
                Nsamples,
                Nβ,
                progress = false,
            ) ≈ m2_threads
        end

        # Compare the results from both backends
        @test m2_serial ≈ m2_threads
    end

    @testset "Exact SRE2" begin
        m2_exact_serial, lost_norm_serial = SRE(ψ, 2; backend = :serial, progress = false)
        m2_exact_threads, lost_norm_threads =
            SRE(ψ, 2; backend = :threads, progress = false)

        # test that we get the same results with both backends
        @test m2_exact_serial ≈ m2_exact_threads
    end

    # Test that Monte Carlo results converge to exact results as Nsamples increases
    # and that the results are improving with more samples
    @testset "Convergence of Monte Carlo" begin
        L = 10
        depth = 4
        ψ = rand_haar(L; depth)

        Nsamples_list = [100, 1000, 10000]
        reps = 100 # number of independent seeds
        diffs = zeros(reps)

        m2_exact, _ = SRE(ψ, 2; backend = :threads, progress = false)

        diff_last = Inf
        for Nsamples in Nsamples_list
            for r = 1:reps
                seed = rand(1:(2^30))  # pick a fresh seed
                Random.seed!(seed)
                m2_mc = MC_SRE(
                    ψ,
                    2;
                    backend = :threads,
                    Nsamples,
                    Nβ = 25,
                    seed,
                    progress = false,
                )
                diffs[r] = abs(m2_mc - m2_exact)
            end

            mean_diff = mean(diffs)

            # Check that the difference is decreasing
            @test mean_diff < diff_last
            diff_last = mean_diff
        end
    end
end
