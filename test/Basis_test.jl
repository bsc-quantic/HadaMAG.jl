@testset "generate_gray_table" begin
    @testset "Binary code" begin
        for n = 0:4
            XTAB, ZT = generate_gray_table(n, 2)

            N = Int(1) << n
            @test length(XTAB) == N
            @test length(ZT) == max(0, N-1)

            @test XTAB[1] == UInt64(0)

            for k = 1:(N-1)
                a, b = XTAB[k], XTAB[k+1]
                diff = a ⊻ b
                @test count_ones(diff) == 1
                @test ZT[k] == trailing_zeros(diff) + 1
                @test b == (UInt64(k) ⊻ (UInt64(k) >> 1))
            end
        end

        # invalid q should throw
        @test_throws AssertionError generate_gray_table(3, 1)
    end

    @testset "General q-code" begin
        # q=3, n=2 → 3^2=9 codes, shape should be (2×9) matrix + 8 pivots
        XT, Z = generate_gray_table(2, 3)
        @test isa(XT, Matrix{Int})
        @test size(XT) == (2, 9)
        @test length(Z) == 8

        # known reflected-Gray 3-ary 2-digit sequence:
        #   [0 1 2  2 1 0  0 1 2;
        #    0 0 0  1 1 1  2 2 2]
        expected = [
            0 1 2 2 1 0 0 1 2;
            0 0 0 1 1 1 2 2 2
        ]
        @test XT == expected

        # Z should label the 1-based digit that changed each step:
        @test Z == [1, 1, 2, 1, 1, 2, 1, 1]

        # a trivial q=4, n=1 test: should just be [0,1,2,3], one row, 3 Zs all 1
        X4, Z4 = generate_gray_table(1, 4)
        @test size(X4) == (1, 4)
        @test vec(X4) == [0, 1, 2, 3]
        @test Z4 == [1, 1, 1]
    end
end
