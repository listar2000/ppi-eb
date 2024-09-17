using Test
include("data/ppi_simulations.jl")

import Test.@testset

@testset "Test get_simulation_α" begin
    n, N, seed = 100, 1000, 2024
    (X, Y), (X̃, Ỹ), true_μ = get_simulation_α(n, N, seed)
    print(true_μ)
    @test size(X) == (n, 3)
    @test size(Y) == (n,)
    @test size(X̃) == (N, 3)
    @test size(Ỹ) == (N,)
    @test isapprox(true_μ, 0.16666, atol=1e-3)
end

@testset "Test get_simulation_β" begin
    n, γ, seed = 100, 0.5, 2024
    N = 20 * n
    (X, Y), (X̃, Ỹ), true_μ = get_simulation_β(n, γ, seed)
    
    @test size(X) == (n, 2)
    @test size(Y) == (n,)
    @test size(X̃) == (N, 2)
    @test size(Ỹ) == (N,)
    @test isapprox(true_μ, 10 * γ / sqrt(6), atol=1e-10)
end