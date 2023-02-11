using DeepPumas
using StableRNGs
using PumasUtilities
using CairoMakie

"""
    true_function(x)

Starting from x=65, increase slowly from 1 up until 2.

# Examples
```jldoctest
julia> true_function.([55, 65, 75, 85, 95])
5-element Vector{Real}:
 1
 1
 1.6321205588285577
 1.8646647167633872
 1.9502129316321362
```
"""
function true_function(x)
    x <= 65 ? 1 : 2 - exp((65 - x) / 10)
end

"""
Plot scatterplots for all input-ouput pairs in `target`.
"""
function plot(target::DeepPumas.FitTarget)
    fig = Figure()
    for i in 1:numinputs(target)
        for j in 1:numoutputs(target)
            scatter(
                fig[i, j],target.x[i, :], target.y[j, :], 
                axis = (
                    xlabel = "covariate[$j]",
                    ylabel = "η[$i]", 
                )
            )
        end
    end
    fig
end

"""
Plot all raw-to-raw scatterplots between matrices `A` and `B`.
"""
function pair_plots(A::Matrix, B::Matrix)

    num_plot_rows, num_plot_cols = size(A)[1], size(B)[1]

    fig = Figure()
    for i in 1:num_plot_rows
        for j in 1:num_plot_cols
            scatter(fig[i, j], A[i, :], B[j, :])
        end
    end
    fig
end

""" 
Pumas model generating data where the deviation of each subject from 
`tvCl` and `tvVc` is deterministically determined by `age` and `weight`.
"""
model_deterministic = @model begin
    @param begin
        tvCl ∈ RealDomain(lower = 0)        # typical value of clearance (L/hr)
        tvVc ∈ RealDomain(lower = 0)        # typical value of central volume of distribution (L)
        Ω ∈ PDiagDomain(2)                  # covariance matrix of random effects (between subject variability)
        σ ∈ RealDomain(lower = 0)           # residual error
    end
    @covariates age weight
    @pre begin
        Cl = tvCl * true_function(age)      # per subject clearance (L/hr)
        Vc = tvVc * true_function(weight)   # per subject volume of central compartment (L)
    end
    @dynamics begin
        Central' = -(Cl / Vc) * Central     # ODE for concentration of drug in plasma (μg/L)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)      # x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

population_deterministic = synthetic_data(
    model_deterministic;
    covariates = (
        age = truncated(Normal(55, 10), 35, Inf),
        weight = truncated(Normal(75, 10), 60, Inf),
    ),
    rng = StableRNG(0),
)

""" 
Pumas model generating data where the deviation of each subject from 
`tvCl` and `tvVc` is independent of `age` and `weight`.
"""
model_independent = @model begin
    @param begin
        tvCl ∈ RealDomain(lower = 0)    # typical value of clearance (L/hr)
        tvVc ∈ RealDomain(lower = 0)    # typical value of central volume of distribution (L)
        Ω ∈ PDiagDomain(2)              # covariance matrix of random effects (between subject variability)
        σ ∈ RealDomain(lower = 0)       # residual error
    end
    @random begin
        η ~ MvNormal(Ω)                 # per subject random effects
    end
    @covariates age weight
    @pre begin
        Cl = tvCl * exp(η[1])           # per subject clearance (L/hr)
        Vc = tvVc * exp(η[2])           # per subject volume of central compartment (L)
    end
    @dynamics begin
        Central' = -(Cl / Vc) * Central # ODE for concentration of drug in plasma (μg/L)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)  # x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

population_independent = synthetic_data(
    model_independent;
    covariates = (
        age = truncated(Normal(55, 10), 35, Inf),
        weight = truncated(Normal(75, 10), 60, Inf),
    ),
    rng = StableRNG(0),
)

target_model_independent_population_deterministic = preprocess(
    model_independent, 
    population_deterministic, 
    init_params(model_independent), 
    FOCE()
)

target_model_independent_population_independent = preprocess(
    model_independent, 
    population_independent, 
    init_params(model_independent), 
    FOCE()
)

pair_plots(
    target_model_independent_population_deterministic.x, 
    target_model_independent_population_deterministic.y
)
pair_plots(
    target_model_independent_population_independent.x,
    target_model_independent_population_independent.y
)

mlp_domain = MLP(
    numinputs(target_model_independent_population_deterministic), 
    16, 16, 
    (numoutputs(target_model_independent_population_deterministic), identity), 
    reg = L2()
)

ho = hyperopt(mlp_domain, target_model_independent_population_deterministic)

mlp_model = ho.ml.ml.model
mlp_parameters = ho.ml.ml.param
y_hat = mlp_model(target_model_independent_population_deterministic.x, mlp_parameters)

pair_plots(target_model_independent_population_deterministic.x, Array(y_hat))
