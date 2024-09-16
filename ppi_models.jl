# Define abstract type for supervised learning models
abstract type SupervisedModel end

# Linear Regression Model
struct LinearRegression <: SupervisedModel
    coefficients::Vector{Float64}
end

function fit(::Type{LinearRegression}, X::Matrix{Float64}, y::Vector{Float64})
    X_aug = hcat(ones(size(X, 1)), X)  # Add intercept term
    coeffs = (X_aug' * X_aug) \ (X_aug' * y)  # Normal equation
    return LinearRegression(coeffs)
end

function predict(model::LinearRegression, X::Matrix{Float64})
    X_aug = hcat(ones(size(X, 1)), X)
    return X_aug * model.coefficients
end

# Quadratic Regression Model
struct QuadraticRegression <: SupervisedModel
    coefficients::Vector{Float64}
end

function fit(::Type{QuadraticRegression}, X::Matrix{Float64}, y::Vector{Float64})
    X_quad = hcat(ones(size(X, 1)), X, X.^2)  # Quadratic terms
    coeffs = (X_quad' * X_quad) \ (X_quad' * y)
    return QuadraticRegression(coeffs)
end

function predict(model::QuadraticRegression, X::Matrix{Float64})
    X_quad = hcat(ones(size(X, 1)), X, X.^2)
    return X_quad * model.coefficients
end

# K-Nearest Neighbors Model
struct KNN <: SupervisedModel
    k::Int
    X_train::Matrix{Float64}
    y_train::Vector{Float64}
end

function fit(::Type{KNN}, X::Matrix{Float64}, y::Vector{Float64}, k::Int)
    return KNN(k, X, y)
end

function predict(model::KNN, X::Matrix{Float64})
    m, _ = size(X)
    y_pred = Vector{Float64}(undef, m)
    for i in 1:m
        distances = sqrt.(sum((model.X_train .- X[i, :]').^2, dims=2))
        idx = sortperm(vec(distances))[1:model.k]  # Get indices of k nearest neighbors
        y_pred[i] = mean(model.y_train[idx])  # For regression, take mean of k nearest neighbors
    end
    return y_pred
end

# calculate the in-sample accuracy
function mse(y_true, y_pred)
    return mean((y_true .- y_pred).^2)
end

# Example usage:

# Generate some toy data
function main()
    X = rand(100, 1)  # 100 samples, 1 feature
    y = 3.0 .* X[:, 1] .+ 2.0 .+ 0.5 .* randn(100)  # Linear relationship with noise

    # Fit and predict with Linear Regression
    lin_model = fit(LinearRegression, X, y)
    y_pred_lin = predict(lin_model, X)

    # Fit and predict with Quadratic Regression
    quad_model = fit(QuadraticRegression, X, y)
    y_pred_quad = predict(quad_model, X)

    # Fit and predict with K-Nearest Neighbors (K=5)
    knn_model = fit(KNN, X, y, 5)
    y_pred_knn = predict(knn_model, X)
    println("MSE with Linear Regression: ", mse(y, y_pred_lin))
    println("MSE with Quadratic Regression: ", mse(y, y_pred_quad))
    println("MSE with KNN: ", mse(y, y_pred_knn))
end 

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end