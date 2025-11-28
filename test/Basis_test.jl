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
            for n = 2:6, P = 1:5
                codes_ref, flips_ref = generate_gray_table(n, 2)

                N = Int(1) << n

                all_codes   = UInt64[]
                total_flips = 0

                for rank = 0:(P-1)
                    xtab, zwhere, code_off, flip_off =
                        HadaMAG.generate_binary_splitted(n, rank, P)

                    # codes partition matches partition_counts
                    code_counts, code_displs = HadaMAG.partition_counts(N, P)
                    @test code_off == code_displs[rank+1]
                    @test length(xtab) == code_counts[rank+1]

                    # flips aligned to codes (by design)
                    @test flip_off == code_off

                    # local adjacency: zwhere labels flip between xtab[j-1] and xtab[j]
                    @test length(zwhere) == max(length(xtab) - 1, 0)
                    @inbounds for j = 2:length(xtab)
                        diff = xtab[j-1] ⊻ xtab[j]
                        expected_flip = trailing_zeros(diff) + 1
                        @test zwhere[j-1] == expected_flip
                    end

                    append!(all_codes, xtab)
                    total_flips += length(zwhere)
                end

                # global coverage of codes
                @test all_codes == codes_ref

                # total flips = N - min(N,P) (no cross-rank transitions)
                @test total_flips == N - min(N, P)
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

        @testset "q-ary splitted Gray code" begin
            # you can tweak ranges as you like
            for n = 2:4, q = 3:5, P = 1:5
                # reference full table and flips
                XTAB_ref, flips_ref = generate_gray_table(n, q)
                N = q^n

                # global storage to reassemble all codes
                all_XTAB = Matrix{Int}(undef, n, N)
                total_flips = 0

                # reference partitioning
                code_counts, code_displs = HadaMAG.partition_counts(N, P)

                for rank = 0:(P-1)
                    local_XTAB, local_flips, code_off, flip_off =
                        HadaMAG.generate_general_splitted(n, q, rank, P)

                    code_cnt = code_counts[rank+1]

                    # codes partition matches partition_counts
                    @test code_off == code_displs[rank+1]
                    @test size(local_XTAB, 2) == code_cnt

                    # flips aligned to codes (by design)
                    @test flip_off == code_off

                    # local adjacency: local_flips labels differences between local_XTAB[:,j-1] and [:,j]
                    @test length(local_flips) == max(code_cnt - 1, 0)

                    @inbounds for j = 2:code_cnt
                        # first differing digit within the slice
                        pos = 0
                        for i = 1:n
                            if local_XTAB[i, j-1] != local_XTAB[i, j]
                                pos = i
                                break
                            end
                        end
                        @test pos != 0
                        @test local_flips[j-1] == pos
                    end

                    # also check against global flips_ref where defined:
                    # local_flips[j] corresponds to global transition k → k+1
                    # with k = code_off + j (1-based)
                    @inbounds for j = 1:length(local_flips)
                        global_k = code_off + j   # 1-based index into flips_ref
                        @test local_flips[j] == flips_ref[global_k]
                    end

                    # copy local codes into the global matrix at the correct positions
                    if code_cnt > 0
                        @inbounds all_XTAB[:, code_off+1 : code_off+code_cnt] = local_XTAB
                    end

                    total_flips += length(local_flips)
                end

                # global coverage of codes
                @test all_XTAB == XTAB_ref

                # total flips = N - min(N, P) (no cross-rank transitions)
                @test total_flips == N - min(N, P)
            end
        end
    end
end
