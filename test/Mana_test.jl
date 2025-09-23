@testset "Mana" begin
    L = 8
    vecvals = randn(ComplexF64, 3^L)
    ψ = StateVec(vecvals; q = 3)
    normalize!(ψ)

    mana_serial = Mana(ψ; backend = :serial, progress = false)
    mana_threads = Mana(ψ; backend = :threads, progress = false)

    # test that we get the same results with both backends
    @test mana_serial ≈ mana_threads
end
