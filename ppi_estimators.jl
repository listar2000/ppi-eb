# Once we obtain the data (both labeled and unlabeled), we can leverage the prediction models
# in the `ppi_estimators.jl` file to predict the outcome for the unlabeled data. We can then
# combine these data to estimate the average response E[Y] using some different estimators.

# Function for compute the vanilla MLE estimator using only the labelled response Y
function vanilla_mle(Y::Vector{Float64}, Ŷ::Vector{Float64}, Ỹ::Vector{Float64})
    return mean(Y)
end

# Function for compute the vanilla PPI estimator 
# Y: the labelled response; Ŷ: the predicted response for the labelled data; 
# Ỹ: the predicted response for the unlabelled data
function vanilla_ppi(Y::Vector{Float64}, Ŷ::Vector{Float64}, Ỹ::Vector{Float64})
    return mean(Ỹ) - mean(Ŷ) + mean(Y)
end

# Function for compute the power-tuned PPI estimator
function power_tuned_ppi(Y::Vector{Float64}, Ŷ::Vector{Float64}, Ỹ::Vector{Float64})
    # compute the optimal power parameter λ
    N, n = length(Ỹ), length(Y)
    # compute the empirical variance of Ŷ and Ỹ (pool them together)
    var_Ỹ = var([Ŷ; Ỹ])
    # compute the empirical covariance between Y and Ŷ
    cov_Y_Ŷ = cov(Y, Ŷ)
    λ = (N / (N + n)) * (cov_Y_Ŷ / var_Ỹ)
    return mean(Ỹ) + (mean(Y) - mean(Ŷ)) * λ
end

# Function for the empirical bayes (EB)-based bias-tradeoff estimator
function eb_bt_ppi(Y::Vector{Float64}, Ŷ::Vector{Float64}, Ỹ::Vector{Float64})
    # compute the prior variance etimator Â
    n = length(Y)
    Â = sum((Y .- Ŷ).^2) / (n - 2) - 1
    λ = Â / (1 + Â)
    # θ̂_labelled = λ * Ŷ + (1 - λ) * Y
    # Ỹ_adjusted = Ỹ + mean(Y .- Ŷ)
    # θ̂_unlabelled = λ * Ỹ + (1 - λ) * Ỹ_adjusted
    return λ * mean(Y) + (1 - λ) * mean(Ỹ)
end