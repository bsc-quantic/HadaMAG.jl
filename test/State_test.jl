using Random
using LinearAlgebra
using JLD2
using NPZ

@testset "StateVec" begin
    ψ = randn(ComplexF64, 2^3)
    sv = StateVec(ψ)
    @test qubits(sv) == 3 # Number of qubits
    @test qudit_dim(sv) == 2 # Qubit dimension
    @test size(sv) == (8,)

    # Not a valid state vector
    bad = randn(ComplexF64, 7)
    @test_throws ArgumentError StateVec(bad)
end

@testset "Haar random state" begin
    ψ = rand_haar(4; depth = 2)
    @test qubits(ψ) == 4
    @test qudit_dim(ψ) == 2
    @test size(ψ) == (16,)
    @test abs(norm(ψ.data) - 1) < 1e-12 # Check normalization
end

@testset "apply_2gate! (StateVec)" begin
    # pick a small 3-qubit system
    n = 3
    dim = 2^n

    # random 3-qubit state
    values = randn(ComplexF64, dim)
    normalize!(values)

    # create StateVec
    ψ = StateVec(values; q = 2)

    @testset "random unitary gate" begin
        U = HadaMAG.haar_random_unitary(2, MersenneTwister(123))

        # test on every distinct qubit pair and test normalization
        for (q1, q2) in ((1, 2), (1, 3), (2, 3))
            ψ_evolved = apply_2gate(ψ, U, q1, q2)
            @test norm(ψ_evolved) ≈ 1.0 # check normalization
        end
    end

    @testset "identity gate" begin
        I = Matrix{ComplexF64}(LinearAlgebra.I, 4, 4)
        ψ_evolved = apply_2gate(ψ, I, 1, 2)

        @test ψ ≈ ψ_evolved
    end
end


@testset "load_state" begin
    @testset "whitespace format" begin
        tmp_file, tmp_io = mktemp()
        close(tmp_io)
        amplitudes = [0.6+0.8im, 1.0-0.5im]
        open(tmp_file, "w") do io
            for a in amplitudes
                println(io, real(a), " ", imag(a))
            end
        end
        sv = load_state(tmp_file; q = 2)
        @test sv.n == 1
        @test sv.q == 2
        @test sv.data == amplitudes
        rm(tmp_file)
    end

    @testset ".jld2" begin
        tmp_file = tempname() * ".jld2"
        data = [0.1+0.2im, -0.3+0.4im, 0.5-0.6im, -0.7-0.8im]
        JLD2.save(tmp_file, "state", data)
        sv = load_state(tmp_file; q = 2)
        @test sv.n == 2
        @test sv.q == 2
        @test sv.data == data
        rm(tmp_file)
    end

    @testset ".npy" begin
        tmp_file = tempname() * ".npy"
        data = [0.1+0.2im, -0.3+0.4im, 0.5-0.6im, -0.7-0.8im]
        NPZ.npzwrite(tmp_file, data)
        sv = load_state(tmp_file; q = 2)
        @test sv.n == 2
        @test sv.q == 2
        @test sv.data == data
        rm(tmp_file)
    end

    @testset ".txt" begin
        tmp_file = tempname() * ".txt"
        data = [0.1+0.2im, -0.3+0.4im, 0.5-0.6im, -0.7-0.8im]
        open(tmp_file, "w") do io
            for a in data
                println(io, real(a), " ", imag(a))
            end
        end
        sv = load_state(tmp_file; q = 2)
        @test sv.n == 2
        @test sv.q == 2
        @test sv.data == data
        rm(tmp_file)
    end
end
