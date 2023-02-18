using DeepPumas
using StableRNGs
using PumasUtilities
using CairoMakie
using JuliaFormatter


# 
# TABLE OF CONTENTS
# 
# 1. MODELS WITHOUT RANDOM EFFECTS
#
# 1.1. A simple UDE model without random effects
# 1.2. Investigate the dynamics of Central captured by the `model_1`
#
# 2. MODELS WITH RANDOM EFFECTS
#
# 2.1. Extend `model_1` adding random effects for Vc
# 2.2. Extend `model_2` adding random effects for CL in the dynamics
# 2.3. Compare `model_3` and `data_model`
# 2.4. In `model_3`, can we switch CL by an "anoynmous" random effect?
# 2.5. In `model_3`, should adding Vc in the `@dynamics` block help?
#

"""
Helper Pumas model to generate synthetic data. The model assumes 
one compartment linear elimination model and IV administration.
(adapted from https://tutorials.pumas.ai/html/PKPDDataAnalysisBook/PK/pk01.html)
"""
data_model = @model begin
    @param begin
        tvCL ∈ RealDomain(lower = 0)    # typical value of CLearance
        tvVc ∈ RealDomain(lower = 0)    # typical value of central volume of distribution
        Ω ∈ PDiagDomain(2)              # covariance matrix of random effects (between subject variability)
        σ ∈ RealDomain(lower = 0)       # residual error
    end
    @random begin
        η ~ MvNormal(Ω)                 # per subject random effects
    end
    @pre begin
        CL = tvCL * exp(η[1])           # per subject CLearance
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

population = synthetic_data(data_model; rng = StableRNG(0))
plotgrid(population[1:4])

# 
# 1. MODELS WITHOUT RANDOM EFFECTS
#
# 1.1. A simple UDE model without random effects
# 1.2. Investigate the dynamics of Central captured by the `model_1`
#

# 1.1. A simple UDE model without random effects

model_1 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity))
        tvVc ∈ RealDomain(lower = 0)
        σ ∈ RealDomain(lower = 0)
    end
    @pre begin
        mlp_ = only ∘ mlp
    end
    @dynamics begin
        Central' = mlp_(Central)
    end
    @derived begin
        cp := @. 1000 * (Central / tvVc)
        dv ~ @. Normal(cp, σ)
    end
end

fpm_1 = fit(
    model_1,
    population,
    init_params(model_1),
    MAP(NaivePooled());
    diffeq_options = (; alg = Rodas5P())
)

pred = predict(fpm_1);
plotgrid(pred[1:4])

# Without random effects (nor covariates), `model_1` can not distinguish patients
# Without random effects, `pred` and `ipred` are identical

# 1.2. Investigate the dynamics of Central captured by the `model_1`

fmlp = only ∘ coef(fpm_1).mlp;
central_prime = [fmlp(central) for central in 0:0.1:5]
lines(
    0:0.1:5,
    central_prime;
    axis = (xlabel = "Central", ylabel = "Central' = mlp(Central)"),
)

# The larger Central is, the fastest is the elimination
# The overall captured dynamics are reasonable

#
# 2. MODELS WITH RANDOM EFFECTS
#
# 2.1. Extend `model_1` adding random effects for Vc
# 2.2. Extend `model_2` adding random effects for CL in the dynamics
# 2.3. Compare `model_3` and `data_model`
# 2.4. In `model_3`, can we switch CL by an "anoynmous" random effect?
# 2.5. In `model_3`, should adding Vc in the `@dynamics` block help?
#

# 2.1. Extend `model_1` adding random effects for Vc

model_2 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity))
        tvVc ∈ RealDomain(lower = 0)
        ω_Vc ∈ RealDomain(lower = 0)
        σ ∈ RealDomain(lower = 0)
    end
    @random begin
        η_Vc ~ Normal(0, ω_Vc)
    end
    @pre begin
        Vc = tvVc * exp(η_Vc)
        mlp_ = only ∘ mlp
    end
    @dynamics begin
        Central' = mlp_(Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)
        dv ~ @. Normal(cp, σ)
    end
end

fpm_2 = fit(
    model_2,
    population,
    init_params(model_2),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 150),
)

pred = predict(fpm_2);
plotgrid(pred[1:4])

# Without covariates, `pred`s are the same for all patients
# Thanks to the random effects on Vc, `ipred`s adapt to each patient

# 2.2. Extend `model_2` adding random effects for CL in the dynamics

model_3 = @model begin
    @param begin
        mlp ∈ MLP(2, 4, 4, (1, identity))
        tvCL ∈ RealDomain(lower = 0)
        tvVc ∈ RealDomain(lower = 0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(lower = 0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        CL = tvCL * exp(η[1])
        Vc = tvVc * exp(η[2])
        mlp_ = only ∘ mlp
    end
    @dynamics begin
        Central' = mlp_(Central, CL)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)
        dv ~ @. Normal(cp, σ)
    end
end

fpm_3 = fit(
    model_3,
    population,
    init_params(model_3),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 150),
)

pred = predict(fpm_3);
plotgrid(pred[1:4])

# Without covariates, `pred`s are the same for all patients
# Thanks to the random effects η[1] and η[2], `ipred`s adapt to each patient

# 2.3. Compare `model_3` and `data_model`
# Hints:
# - Should `model_2` be able to produce good predictions? Why?
# - What are the differences between the dynamics in the two models?

# 2.4. In `model_3`, can we switch CL by an "anoynmous" random effect?
# TODO: This model doesn't train properly always. It seems that having η_ after Central helps?
model_4 = @model begin
    @param begin
        mlp ∈ MLP(2, 4, 4, (1, identity))
        tvVc ∈ RealDomain(lower = 0)
        ω_Vc ∈ RealDomain(lower = 0)
        σ ∈ RealDomain(lower = 0)
    end
    @random begin
        η ~ Normal(0, 1)
        η_Vc ~ Normal(0, ω_Vc)
    end
    @pre begin
        η_ = η
        Vc = tvVc * exp(η_Vc)
        mlp_ = only ∘ mlp
    end
    @dynamics begin
        Central' = mlp_(Central, η_)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)
        dv ~ @. Normal(cp, σ)
    end
end

fpm_4 = fit(
    model_4,
    population,
    init_params(model_4),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 150),
)

pred = predict(fpm_4);
plotgrid(pred[1:4])

# Without covariates, `pred`s are the same for all patients
# Thanks to the random effects η and η_Vc, `ipred`s adapt to each patient
# Calling the ranfom effect η, η[1], or η_CL does not change its function

# 2.5. In `model_3`, should adding Vc in the `@dynamics` block help?

model_5 = @model begin
    @param begin
        mlp ∈ MLP(3, 4, 4, (1, identity))
        tvCL ∈ RealDomain(lower = 0)
        tvVc ∈ RealDomain(lower = 0)
        Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(lower = 0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        CL = tvCL * exp(η[1])
        Vc = tvVc * exp(η[2])
        mlp_ = only ∘ mlp
    end
    @dynamics begin
        Central' = mlp_(Central, CL, Vc)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)
        dv ~ @. Normal(cp, σ)
    end
end

fpm_5 = fit(
    model_5,
    population,
    init_params(model_5),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 150),
)

pred = predict(fpm_5);
plotgrid(pred[1:4])

pred = predict(fpm_3);
plotgrid(pred[1:4])

# Vc is important in the `@derived` block to model concentration
# However, Vc is redundant in the `@dynamics` block because CL, 
# through its random effect η[1], is already informing