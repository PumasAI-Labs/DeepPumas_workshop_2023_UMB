using DeepPumas
using Pumas
using StableRNGs
using PumasUtilities
using CairoMakie

"""
    true_function(x)

Starting from x=65, increase slowly up until 1.

# Examples
```jldoctest
julia> true_function.([55, 65, 75, 85, 95])
5-element Vector{Real}:
 0
 0
 0.6321205588285577
 0.8646647167633873
 0.950212931632136
```
"""
function true_function(x)
    x <= 65 ? 0 : 1 - exp((65 - x) / 10)
end

""" Data generating model with covariate `Age` affecting `Cl`earance. """
data_model = @model begin
    @param begin
        tvCl ∈ RealDomain(lower = 0)    # typical value of clearance (L/hr)
        tvVc ∈ RealDomain(lower = 0)    # typical value of central volume of distribution (L)
        Ω ∈ PDiagDomain(2)              # covariance matrix of random effects (between subject variability)
        σ ∈ RealDomain(lower = 0)       # residual error
    end
    @random begin
        η ~ MvNormal(Ω)                 # per subject random effects
    end
    @covariates Age
    @pre begin
        Cl = tvCl * exp(η[1]) + true_function(Age)  # per subject clearance (L/hr)
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

population = synthetic_data(
    data_model;
    covariates = (Age = truncated(Normal(65, 10), 35, Inf),),
    rng = StableRNG(0),
)

preds = predict(data_model, population[1:4], init_params(data_model))
plotgrid(preds)

""" Model with a universal differential equation. """
ude_model = @model begin
    @param begin
        mlp ∈ MLP(3, 4, 4, (1, identity); reg = L2(1.0))
        tvCl ∈ RealDomain(lower = 0)
        tvVc ∈ RealDomain(lower = 0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(lower = 0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        Cl = tvCl * exp(η[1])
        Vc = tvVc * exp(η[2])
        mlp_ = only ∘ mlp
    end
    @dynamics begin
        Central' = mlp_(Cl, Vc, Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)
        dv ~ @. Normal(cp, σ)
    end
end

fitted_ude_model = fit(
    ude_model,
    population,
    init_params(ude_model),
    MAP(FOCE());
    diffeq_options = (alg = Rodas5P(),),
    optim_options = (iterations = 150,),
)

pred = predict(ude_model, population[1:4], coef(fitted_ude_model); obstimes = 0:0.1:10);
pred = predict(ude_model, population[1:4], coef(fitted_ude_model); obstimes = 0:0.1:10);
plotgrid(pred[1:4])