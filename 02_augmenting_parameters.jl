using DeepPumas
using DeepPumas.SimpleChains
# using PumasPlots
using PumasPlots.CairoMakie
using StableRNGs
# using PumasUtilities
# using CairoMakie
include("utils/utils.jl")

# TABLE OF CONTENTS
# 1. Sample population where individual parameters are fixed effects
# 2. Sample population where individual parameters are random effects
# 3. Study the relationship between covariates and random effects
# 4. Predict individual parameters as a function of covariates 
# 5. DeepPumas model augmentation


# 1. Sample population where individual parameters are fixed effects

"""
Helper Pumas model to generate synthetic data. The deviation of each 
subject from `tvCL` and `tvVc` is deterministically determined by the 
covariates `age` and `weight`.
"""
model_deterministic = @model begin
    @param begin
        tvCL ∈ RealDomain(lower = 0)    # typical value of clearance
        tvVc ∈ RealDomain(lower = 0)    # typical value of central volume of distribution
        σ ∈ RealDomain(lower = 0)       # residual error
    end
    @covariates age weight
    @pre begin
        CL = tvCL * saturating_function(age)    # per subject clearance
        Vc = tvVc * saturating_function(weight) # per subject volume of central compartment
    end
    @dynamics begin
        Central' = -(CL / Vc) * Central # ODE for concentration of drug in plasma
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)  # x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

pop_deterministic = synthetic_data(
    model_deterministic;
    covariates = (
        age = truncated(Normal(55, 10), 35, Inf),
        weight = truncated(Normal(75, 10), 60, Inf),
    ),
    rng = StableRNG(0),
)

# 2. Sample population where individual parameters are random effects

"""
Helper Pumas model to generate synthetic data. The deviation of each 
subject from `tvCL` and `tvVc` is random, and independent from the 
covariates `age` and `weight`.
"""
model_random = @model begin
    @param begin
        tvCL ∈ RealDomain(lower = 0)    # typical value of clearance
        tvVc ∈ RealDomain(lower = 0)    # typical value of central volume of distribution
        Ω ∈ PDiagDomain(2)              # covariance matrix of random effects (between subject variability)
        σ ∈ RealDomain(lower = 0)       # residual error
    end
    @random begin
        η ~ MvNormal(Ω)                 # per subject random effects
    end
    @covariates age weight
    @pre begin
        CL = tvCL * exp(η[1])           # per subject clearance
        Vc = tvVc * exp(η[2])           # per subject volume of central compartment
    end
    @dynamics begin
        Central' = -(CL / Vc) * Central # ODE for concentration of drug in plasma
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)  # x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

pop_random = synthetic_data(
    model_random;
    covariates = (
        age = truncated(Normal(55, 10), 35, Inf),
        weight = truncated(Normal(75, 10), 60, Inf),
    ),
    rng = StableRNG(0),
)

# 3. Study the relationship between covariates and random effects
# Hints: 
#   - `DeepPumas.preprocess` computes EBEs of random effects and
#      prepares a mapping of covariates to those EBEs
#   - Use `DeepPumas.preprocess` on `model_random`, both for 
#     `pop_deterministic` and for `pop_random`, because the data 
#     generating process is unknown to us.
#   - The function `pair_plots` plots pairwise scatterplots.

cov2randeff_deterministic =
    preprocess(model_random, pop_deterministic, init_params(model_random), FOCE())
pair_plots(cov2randeff_deterministic.x, cov2randeff_deterministic.y)

cov2randeff_random = preprocess(model_random, pop_random, init_params(model_random), FOCE())
pair_plots(cov2randeff_random.x, cov2randeff_random.y)

# 4. Predict individual parameters as a function of covariates 

mlp = MLP(2, 8, (2, identity))
fmlp_deterministic =
    fit(mlp, cov2randeff_deterministic; optim_options = (; optim_alg = SimpleChains.ADAM()))
ŷ = Array(mlp.model(cov2randeff_deterministic.x, fmlp.ml.ml.param))
pair_plots(
    cov2randeff_deterministic.y,
    ŷ,
    xlabels = ["y₁", "y₂"],
    ylabels = ["ŷ₁", "ŷ₂"],
)
pair_plots(
    cov2randeff_deterministic.x,
    ŷ,
    xlabels = ["age", "weight"],
    ylabels = ["ŷ₁", "ŷ₂"],
)

mlp = MLP(2, (64, relu), (64, relu), (2, identity))
fmlp = fit(mlp, cov2randeff_random; optim_options = (; optim_alg = SimpleChains.ADAM()))
ŷ = mlp.model(cov2randeff_random.x, fmlp.ml.ml.param)
pair_plots(cov2randeff_random.y, ŷ, xlabels = ["y₁", "y₂"], ylabels = ["ŷ₁", "ŷ₂"])
pair_plots(cov2randeff_random.x, ŷ, xlabels = ["age", "weight"], ylabels = ["ŷ₁", "ŷ₂"])

# 5. DeepPumas model augmentation

# fpm_deterministic = fit(
#   model_deterministic,
#   pop_deterministic,
#   init_params(model_deterministic),
#   MAP(NaivePooled())
# )

# pred = predict(model_deterministic, pop_deterministic, coef(fpm_deterministic));
# plotgrid(pred[1:8])

# fpm_random = fit(
#   model_random,
#   pop_random,
#   init_params(model_random),
#   MAP(NaivePooled())
# )

# pred = predict(model_random, pop_random, coef(fpm_random));
# plotgrid(pred[1:8])


fpm = fit(
  model_random,
  pop_deterministic,
  init_params(model_random),
  MAP(NaivePooled())
)

pred = predict(model_random, pop_deterministic, coef(fpm));
plotgrid(pred[1:8])

augmented_model = augment(fpm, cov2randeff_deterministic)

# The `init_params` of an augmented model is the combination of the best parameters of the
# FittedPumasModel and the fitted machine learning model.
p_covs = init_params(augmented_model)
covariate_pred = predict(augmented_model, testpop, p_covs; obstimes)

plotgrid(true_pred; pred = (; label="Best possible pred", color=(:black, 0.5)), ipred = false)
plotgrid!(covariate_pred; pred = (; linestyle=:dash))

# TODO
# ADD HERE WORKFLOW AS IN DEMO30 or https://github.com/PumasAI/DeepPumas-Workshop/blob/main/code/02-model_identification.jl
