using DeepPumas
using DeepPumas.SimpleChains
using PumasPlots.CairoMakie
using StableRNGs
include("utils/utils.jl")

#
# TABLE OF CONTENTS
#
# 1. SYNTHETIC DATA GENERATION
# 1.1. Sample synthetic data where individual parameters are deterministic
# 1.2. Exercise: Reason about the predictions of the data-generating model
#
# 2. PUMAS: MODEL THE POPULATION
# 2.1. Define and fit a Pumas model
# 2.2. Exercise: Reason about the predictions of the Pumas model
#
# 3. DEEPPUMAS: AUGMENT THE MODEL WITH COVARIATES
# 3.1. Model the relationship between covariates and random effects
# 3.2. Augment the model with covariates and reason about its predictions
# 3.3. Continue fitting the augmented model
#

#
# 1. SYNTHETIC DATA GENERATION
# 1.1. Sample synthetic data where individual parameters are deterministic
# 1.2. Exercise: Reason about the predictions of the data-generating model
#

# 1.1. Sample synthetic data where individual parameters are deterministic

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
        CL = tvCL * exp(saturating_function(age))    # per subject clearance
        Vc = tvVc * exp(saturating_function(weight)) # per subject volume of central compartment
    end
    @dynamics begin
        Central' = -(CL / Vc) * Central # ODE for concentration of drug in plasma
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)  # x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

population = synthetic_data(
    model_deterministic;
    covariates = (
        age = truncated(Normal(55, 10), 35, Inf),
        weight = truncated(Normal(75, 10), 60, Inf),
    ),
    rng = StableRNG(0),
)

# 1.2. Exercise: Reason about the predictions of the data-generating model

pred_true = predict(model_deterministic, population, init_params(model_deterministic));
plotgrid(pred_true[1:8]; pred=(; label="Pred (data-generating model)"), ipred=false)

#
# 2. PUMAS: MODEL THE POPULATION
# 2.1. Define and fit a Pumas model
# 2.2. Exercise: Reason about the predictions of the Pumas model
#

# 2.1. Define and fit a Pumas model

model = @model begin
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

fpm = fit(
  model,
  population,
  init_params(model),
  MAP(FOCE())  # TODO: NaivePooled gives undetermined Ω but FOCE doesn't. Why?
)

# 2.2. Exercise: Reason about the predictions of the Pumas model

pred = predict(model, population, init_params(model));
plotgrid!(pred[1:8]; pred=(; label="Pred (model)", color=:red), ipred=false)

#
# 3. DEEPPUMAS: AUGMENT THE MODEL WITH COVARIATES
# 3.1. Model the relationship between covariates and random effects
# 3.2. Augment the model with covariates and reason about its predictions
# 3.3. Continue fitting the augmented model
#

# 3.1. Model the relationship between covariates and random effects
# Hints: 
#   - `DeepPumas.preprocess` computes EBEs of random effects and
#      prepares a mapping of covariates to those EBEs
#   - The function `pair_plots` plots pairwise scatterplots

cov2randeff =
    preprocess(model, population, init_params(model), FOCE())
pair_plots(cov2randeff.x, cov2randeff.y, xlabels=["age", "weight"], ylabels=["η₁", "η₂"])

mlp = MLP(2, 8, (2, identity))
fmlp =
    fit(mlp, cov2randeff; optim_options = (; optim_alg = SimpleChains.ADAM()))
η̂ = Array(mlp.model(cov2randeff.x, fmlp.ml.ml.param));
pair_plots(
    cov2randeff.y,
    η̂,
    xlabels = ["η₁", "η₂"],
    ylabels = ["η̂₁(age, weight)", "η̂₂(age, weight)"],
)
pair_plots(
    cov2randeff.x,
    η̂,
    xlabels = ["age", "weight"],
    ylabels = ["η̂₁(age, weight)", "η̂₂(age, weight)"],
)

# 3.2. Augment the model with covariates and reason about its predictions
# Hints: 
#   - `DeepPumas.augment` augments a fitted Pumas model with a fitted ML model
#   - The `init_params` of an augmented model are the fitted coefficients of 
#     the Pumas model and the fitted coefficients of the ML model

model_augmented = augment(fpm, fmlp)

pred_augmented = predict(model_augmented, population, init_params(model_augmented));
# pred_augmented = predict(model_augmented, population, merge(init_params(model_augmented), init_params(model_deterministic)));

plotgrid(pred_true[1:8]; pred=(; label="Pred (data-generating model)"), ipred=false)
plotgrid!(pred[1:8]; pred=(; label="Pred (model)", color=:red), ipred=false)
plotgrid!(pred_augmented[1:8]; pred=(; label="Pred (initial params augmented model)", color=:green), ipred=false)

# 3.4. Continue fitting the augmented model

# DomainError with Inf. Ideas?
fapm = fit(
    model_augmented,
    population,
    init_params(model_augmented),
    MAP(FOCE())
)

plotgrid(pred_true[1:8]; pred=(; label="Pred (data-generating model)"), ipred=false)
plotgrid!(pred[1:8]; pred=(; label="Pred (model)", color=:red), ipred=false)
plotgrid!(pred_augmented[1:8]; pred=(; label="Pred (augmented model)", color=:green), ipred=false)