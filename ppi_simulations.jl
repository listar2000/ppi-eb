using Random, Distributions, LinearAlgebra

# simulation study mentioned in the note Example 2
function get_simulation_α(n::Int = 100, N::Int = 2000, seed::Int = 2024)
    Random.seed!(seed)
    d = 3
    σ² = 0.1
    m(x) = 2x[1] - x[2]^2 + x[3] - 1
    X = rand(n, d)
    Y = m.(eachrow(X)) + sqrt(σ²) .* randn(n)
    X̃ = rand(N, d)
    # \hat X as variable name
    Ỹ = m.(eachrow(X̃)) + sqrt(σ²) .* randn(N)
    true_μ = 2 * 0.5 - (1/12 + 0.5^2) + 0.5 - 1
    return (X, Y), (X̃, Ỹ), true_μ
end

# simulation study mentioned in Miao et al. (2024)
function get_simulation_β(n::Int = 250, γ::Float64 = 0.5, seed::Int = 2024)
    Random.seed!(seed)
    # Unlabeled data is 20 times more than labeled data
    N = 20 * n  
    # Covariates X are drawn from a multivariate normal distribution
    Σ = I(2)  # Identity covariance matrix for 2 dimensions
    X = Matrix(rand(MvNormal([0.0, 0.0], Σ), n)')  # (n, 2) matrix where X[i, :] is [X1i, X2i]
    
    # Generate the outcome Y based on the provided formula
    Y = 5 * γ .* (X[:, 1] + X[:, 2] + X[:, 1].^2 + X[:, 2].^2 + X[:, 1] .* X[:, 2]) / sqrt(6) +
        randn(n) .* sqrt((5 * sqrt(1 - γ^2)))
    
    # Unlabeled data (N times more than labeled data)
    X̃ = Matrix(rand(MvNormal([0.0, 0.0], Σ), N)')  # (N, 2) unlabeled samples
    Ỹ = 5 * γ .* (X̃[:, 1] + X̃[:, 2] + X̃[:, 1].^2 + X̃[:, 2].^2 + X̃[:, 1] .* X̃[:, 2]) / sqrt(6) +
        randn(N) .* sqrt((5 * sqrt(1 - γ^2)))
        
    true_μ = 10 * γ / sqrt(6)
    return (X, Y), (X̃, Ỹ), true_μ
end

# using Plots

# # Generate data using get_simulation_α function
# data_α, _, _ = get_simulation_α(1000, 1000)
# Y_α = data_α[2]

# # Generate data using get_simulation_β function
# data_β, _, _ = get_simulation_β(1000, 1000, 0.5)
# Y_β = data_β[2]

# # Plot histogram for Simulation α
# histogram(Y_α, xlabel="Y", ylabel="Frequency", title="Histogram of Simulation α")
# savefig("histogram_alpha.png")

# # Plot histogram for Simulation β
# histogram(Y_β, xlabel="Y", ylabel="Frequency", title="Histogram of Simulation β")
# savefig("histogram_beta.png")


