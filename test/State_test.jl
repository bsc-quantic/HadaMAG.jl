using Test
using Random
using LinearAlgebra
using JLD2
using NPZ

using HadaMAG

@testset "StateVec" begin
    ψ = randn(ComplexF64, 2^3)
    sv = StateVec(ψ)
    @test sv.n == 3 # Number of qubits
    @test sv.q == 2 # Qubit dimension
    @test size(sv) == (8,)

    # Not a valid state vector
    bad = randn(ComplexF64, 7)
    @test_throws ArgumentError StateVec(bad)
end

@testset "Haar random state" begin
    sv = rand_haar(4; q=2)
    @test sv.n == 4
    @test sv.q == 2
    @test size(sv) == (16,)
    @test abs(norm(sv.data) - 1) < 1e-12 # Check normalization
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
        sv = load_state(tmp_file; q=2)
        @test sv.n == 1
        @test sv.q == 2
        @test sv.data == amplitudes
        rm(tmp_file)
    end

    @testset ".jld2" begin
        tmp_file = tempname() * ".jld2"
        data = [0.1+0.2im, -0.3+0.4im, 0.5-0.6im, -0.7-0.8im]
        JLD2.save(tmp_file, "state", data)
        sv = load_state(tmp_file; q=2)
        @test sv.n == 2
        @test sv.q == 2
        @test sv.data == data
        rm(tmp_file)
    end

    @testset ".npy" begin
        tmp_file = tempname() * ".npy"
        data = [0.1+0.2im, -0.3+0.4im, 0.5-0.6im, -0.7-0.8im]
        NPZ.npzwrite(tmp_file, data)
        sv = load_state(tmp_file; q=2)
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
        sv = load_state(tmp_file; q=2)
        @test sv.n == 2
        @test sv.q == 2
        @test sv.data == data
        rm(tmp_file)
    end
end