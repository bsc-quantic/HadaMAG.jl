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
