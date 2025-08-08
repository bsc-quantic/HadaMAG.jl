@testset "generate_gray_table" begin

    @testset "Binary code" begin
        for n = 0:4
            codes, flips = generate_gray_table(n, 2)

            N = Int(1) << n
            @test length(codes) == N
            @test length(flips) == max(0, N-1)

            # first code always zero
            @test codes[1] == UInt64(0)

            # each step flips exactly one bit, and matches i⊻(i>>1)
            for k = 1:(N-1)
                a, b = codes[k], codes[k+1]
                diff = a ⊻ b
                @test count_ones(diff) == 1
                @test flips[k] == trailing_zeros(diff) + 1 # flips are 1-based !
                @test b == (UInt64(k) ⊻ (UInt64(k) >> 1))
            end
        end

        # invalid base should throw
        @test_throws AssertionError generate_gray_table(3, 1)

        @testset "Splitted code" begin
            # Test that concatenating the results of `generate_binary_splitted`
            # for all ranks gives the same result as `generate_binary`

            # run tests for a variety of n and P
            for n in 2:6, P in 1:5
                # reference full table
                codes_ref, flips_ref = generate_gray_table(n, 2)

                # collect all MPI‑local pieces
                all_codes = UInt64[]
                all_flips = Int[]

                for rank in 0:P-1
                    xtab, zwhere, code_off, flip_off =
                    HadaMAG.generate_binary_splitted(n, rank, P)
                                        append!(all_codes, xtab)
                    append!(all_flips, zwhere)

                    # test the code‐offset against the codes partition
                    N = Int(1) << n
                    code_counts, code_displs = HadaMAG.partition_counts(N,    P)
                    flip_counts, flip_displs = HadaMAG.partition_counts(N-1,  P)

                    @test code_off == code_displs[rank+1]
                    @test flip_off == flip_displs[rank+1]
                end

                @test all_codes == codes_ref
                @test all_flips == flips_ref
            end
        end

        @testset "Splitted code" begin
            for n in 2:6, P in 1:5
                # reference full table
                codes_ref, flips_ref = generate_gray_table(n, 2)

                all_codes = UInt64[]
                all_flips = Int[]

                for rank in 0:P-1
                    # now returns (xtab, zwhere, code_off, flip_off)
                    xtab, zwhere, code_off, flip_off =
                        HadaMAG.generate_binary_splitted(n, rank, P)

                    append!(all_codes, xtab)
                    append!(all_flips, zwhere)

                    # test the code‐offset against the codes partition
                    N = Int(1) << n # 2^n
                    code_counts, code_displs = HadaMAG.partition_counts(N,    P)
                    flip_counts, flip_displs = HadaMAG.partition_counts(N-1,  P)

                    @test code_off == code_displs[rank+1]
                    @test flip_off == flip_displs[rank+1]
                end

                @test all_codes == codes_ref
                @test all_flips == flips_ref
            end
        end

    end

    @testset "General q-code" begin
        for (n, q) in ((2, 3), (1, 4))
            XT, Z = generate_gray_table(n, q)
            N = q^n

            @test isa(XT, Matrix{Int})
            @test size(XT) == (n, N)
            @test length(Z) == max(0, N-1)

            # each column matches integer_to_gray(k-1, q, n)
            for k = 1:N
                @test XT[:, k] == HadaMAG.integer_to_gray(k-1, q, n)
            end

            # each step flips exactly one digit, and Z labels it
            for k = 1:(N-1)
                @test Z[k] == findfirst(i -> XT[i, k] != XT[i, k+1], 1:n)
            end
        end
    end

end
