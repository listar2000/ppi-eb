# this file is the actual place where we combine the simulations, predictive models, and estimators
include("ppi_simulations.jl")
include("ppi_models.jl")
include("ppi_estimators.jl")

ESTIMATORS = [vanilla_mle, vanilla_ppi, power_tuned_ppi, eb_bt_ppi]
# Define the estimator names for labeling the x-axis
estimator_names = ["vanilla_mle", "vanilla_ppi", "power_tuned_ppi", "eb_bt_ppi"]

# a generic function that takes in simulation function, prediction model, repeat trials
# the function should take each estimator, compute the MSE with the ground true value
# and return a matrix of shape (n_estimators, n_trials).
function run_simulation_α(model::Type{<:SupervisedModel}, n_trials::Int, n::Int, N::Int, seed::Int)
    n_estimators = length(ESTIMATORS)
    mse_results = zeros(n_estimators, n_trials)  # Assuming n_estimators is defined

    (X_holdout, Y_holdout), _, _ = get_simulation_α(n, 1, seed)
    # if K-NN, supply the value of k
    if model == KNN
        model_instance = fit(model, X_holdout, Y_holdout, 5)
    else
        model_instance = fit(model, X_holdout, Y_holdout)
    end
    for j in 1:n_estimators
        estimator = ESTIMATORS[j]  # Assuming estimators is defined and contains all estimators
        for i in 1:n_trials
            (X, Y), (X̃, Ỹ), true_μ = get_simulation_α(n, N, seed * j + i)
            # predict the outcome for the unlabelled data
            Ỹ_pred = predict(model_instance, X̃)
            # predict the outcome for the labelled data
            Ŷ = predict(model_instance, X)
            # compute the estimator
            θ̂ = estimator(Y, Ŷ, Ỹ_pred)
            mse_results[j, i] = (true_μ - θ̂)^2
        end
    end
    return mse_results
end

function run_simulation_β(model::Type{<:SupervisedModel}, n_trials::Int, n::Int, γ::Float64, seed::Int)
    n_estimators = length(ESTIMATORS)
    mse_results = zeros(n_estimators, n_trials)  # Assuming n_estimators is defined

    (X_holdout, Y_holdout), _, _ = get_simulation_β(500, γ, seed)
    # if K-NN, supply the value of k
    if model == KNN
        model_instance = fit(model, X_holdout, Y_holdout, 5)
    else
        model_instance = fit(model, X_holdout, Y_holdout)
    end
    for j in 1:n_estimators
        estimator = ESTIMATORS[j]  # Assuming estimators is defined and contains all estimators
        for i in 1:n_trials
            (X, Y), (X̃, Ỹ), true_μ = get_simulation_β(n, γ, seed * j + i)
            # predict the outcome for the unlabelled data
            Ỹ_pred = predict(model_instance, X̃)
            # predict the outcome for the labelled data
            Ŷ = predict(model_instance, X)
            # compute the estimator
            θ̂ = estimator(Y, Ŷ, Ỹ_pred)
            mse_results[j, i] = (true_μ - θ̂)^2
        end
    end
    return mse_results
end

using Plots, StatsPlots, Measures, Statistics

# Function to plot boxplots for a given result matrix
function plot_mse_boxplot(result_matrix, model_name)
    n_estimators, n_trials = size(result_matrix)
    group_labels = repeat(estimator_names, inner=n_trials)  # Labels for the x-axis
    mse_data = vec(result_matrix')  # Flatten the matrix column-wise
    boxplot(group_labels, mse_data, xlabel="Estimators", ylabel="MSE", title=model_name)
end

function run_all_α()
    linear_results = run_simulation_α(LinearRegression, 100, 100, 2000, 2024)
    # print the mean of each estimator
    println("Linear Regression Results:")
    for i in 1:3
        println("Estimator: ", ESTIMATORS[i])
        println("Mean MSE: ", mean(linear_results[i, :]))
    end
    quad_results = run_simulation_α(QuadraticRegression, 100, 100, 2000, 2024)
    println("Quadratic Regression Results:")
    for i in 1:3
        println("Estimator: ", ESTIMATORS[i])
        println("Mean MSE: ", mean(quad_results[i, :]))
    end
    knn_results = run_simulation_α(KNN, 100, 100, 2000, 2024)
    println("K-Nearest Neighbors Results:")
    for i in 1:3
        println("Estimator: ", ESTIMATORS[i])
        println("Mean MSE: ", mean(knn_results[i, :]))
    end
    # Plot for Linear Regression Results
    p1 = plot_mse_boxplot(linear_results, "Linear Regression")

    # Plot for Quadratic Regression Results
    p2 = plot_mse_boxplot(quad_results, "Quadratic Regression")

    # Plot for KNN Results
    p3 = plot_mse_boxplot(knn_results, "K-Nearest Neighbors")

    # Combine the plots into one layout for comparison
    p = plot(p1, p2, p3, layout=(1, 3), size=(1200, 500), left_margin = 10mm, right_margin = 10mm, top_margin = 20mm, bottom_margin = 10mm)
    # save this plot with high dpi and size that's good for use in overleaf paper
    savefig(p, "simulation_α.pdf")
end

# Complete run_all_β function
function run_all_β(n_trials::Int, n::Int, seed::Int, model::Type{<:SupervisedModel})
    # γ values from 0.5 to 1.0 in steps of 0.1
    γ_values = 0.5:0.1:1.0
    
    # Initialize arrays to store results for each estimator across γ values
    mean_mse = Dict{String, Vector{Float64}}()
    q25_mse = Dict{String, Vector{Float64}}()
    q75_mse = Dict{String, Vector{Float64}}()

    # Initialize for each estimator
    for estimator in estimator_names
        mean_mse[estimator] = []
        q25_mse[estimator] = []
        q75_mse[estimator] = []
    end

    # Run simulations for each γ value
    for γ in γ_values
        println("Running simulations for γ = $γ")

        # Run simulations for each model and each estimator
        mse_results = run_simulation_β(model, n_trials, n, γ, seed)
        
        # For each estimator, compute mean, 25th, and 75th quantiles
        for j in 1:length(estimator_names)
            mse_for_estimator = mse_results[j, :]
            append!(mean_mse[estimator_names[j]], mean(mse_for_estimator))
            append!(q25_mse[estimator_names[j]], quantile(mse_for_estimator, 0.25))
            append!(q75_mse[estimator_names[j]], quantile(mse_for_estimator, 0.75))
        end
    end

    γ_values_array = collect(γ_values)  # Convert range to array

    p = plot()
    # Plot results
    for estimator in estimator_names
        p = plot!(
            p,
            γ_values_array,
            mean_mse[estimator],
            ribbon=(q25_mse[estimator], q75_mse[estimator]),  # Add quantiles as ribbon
            label=string(estimator),
            xlabel="γ",
            ylabel="Mean MSE",
            title="MSE Across Different γ Values",
            fillalpha = 0.3,
            lw=2
        )
    end
    # savefig(p, "EB_β_$model.png")  # Save as PNG
    savefig(p, "EB_β_$model.pdf")  # Save as PDF for Overleaf
end

run_all_β(100, 500, 1234, QuadraticRegression)