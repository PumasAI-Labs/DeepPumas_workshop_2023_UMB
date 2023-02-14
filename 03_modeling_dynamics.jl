using DeepPumas
using StableRNGs
using PumasUtilities
using CairoMakie


"""
Example derived from https://tutorials.pumas.ai/html/PKPDDataAnalysisBook/PK/pk01.html

Background

        - Structural model        - One compartment linear elimination
        - Route of administration - IV bolus
        - Dosage Regimen          - 10 mg IV
        - Number of Subjects      - 4
"""

""" Data generating model. """
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

population = synthetic_data(data_model; rng = StableRNG(0))
preds = predict(data_model, population[1:4], init_params(data_model))
plotgrid(preds)

""" Model with a universal differential equation. """
ude_model = @model begin
    @param begin
        mlp ∈ MLP(3, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
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
        mlp_ = only ∘ mlp  # technical
    end
    @dynamics begin
        Central' = mlp_(Cl, Vc, Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)# x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end


# discuss how CL and Vc are actually interchangeable if using Central' = mlp_(CL, Vc, Central)[1]

fpm = fit(
    ude_model,
    population,
    init_params(ude_model),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 150),
)

pred = predict(ude_model, population[1:4], coef(fpm); obstimes = 0:0.1:10);
plotgrid(pred[1:4])

#  DISCUSS THIS Model
# IDENTIFIABILITY ISSUES
# EXAMINING THE MLP ON ITS OWN, FOR EXAMPLE PLOTTING
fmlp = coef(fpm).mlp
fmlp(1, 1, 1)
map(fmlp, (1, 1, 0:0.1:10))
# Central' = -(Cl / Vc) * Central

xrange = 0:0.01:1
res = map(xrange) do x
    fmlp(1.0, 1.0, x)[1]
end
lines(xrange, res)

-(coef(fpm).tvCl / coef(fpm).tvVc)

# IT IS POSSIBLE TO resume training WITH PREVIOUS PARAMS
fpm = fit(
    ude_model,
    population,
    coef(fpm),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 75),
)

pred = predict(ude_model, population[1:4], coef(fpm); obstimes = 0:0.1:10);
plotgrid(pred)
plotgrid(pred; pred = false)


# resample_params = true
# nrepeats = 10
# for _ in 1:nrepeats
#   global fpm = fit(
#     model_ts,
#     sample(trainpop_TS, 30; replace=false),
#     resample_params ? sample_params(model_ts) : coef(fpm),
#     MAP(FOCE());
#     optim_options = (; time_limit=2*60, iterations=20),
#     diffeq_options = (; alg = Rodas5P()),
#     checkidentification=false
#   )
#   display(plotgrid(predict(model_ts, testpop_TS[1:24], coef(fpm))))
#   resample_params = false
# end
# plotgrid(predict(model_ts, testpop_TS[1:24], coef(fpm); obstimes))

# EXAMPLE WITH UDE HAVING ONLY CENTRAL AND NO RANDOM EFFECTS 
# THIS MEANS THAT THERE ISN'T ANY PERSONALIZATION AND PREDICTIONS
# ARE EXACTLY THE SAME FOR ALL PATIENTS
ude_model_2 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
        # tvCl ∈ RealDomain(lower = 0)
        tvVc ∈ RealDomain(lower = 0)
        # Ω ∈ PDiagDomain(2)
        σ ∈ RealDomain(lower = 0)
    end
    @pre begin
        mlp_ = only ∘ mlp  # technical
    end
    @dynamics begin
        Central' = mlp_(Central)  # dCentral/dt = mlp_(Central)
    end
    @derived begin
        cp := @. 1000 * (Central / tvVc)# x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

fpm2 = fit(
    ude_model_2,
    population,
    init_params(ude_model_2),
    MAP(NaivePooled());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 500),
)

pred = predict(ude_model_2, population[1:4], coef(fpm2); obstimes = 0:0.1:10);
plotgrid(pred)


fmlp2 = coef(fpm2).mlp

xrange = 0:0.01:1
res = map(xrange) do x
    fmlp2(x)[1]
end
lines(xrange, res)

# EXAMPLE WITH UDE HAVING ONLY CENTRAL BUT ADDING RANDOM EFFECTS 

ude_model_3 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
        # tvCl ∈ RealDomain(lower = 0)
        tvVc ∈ RealDomain(lower = 0)
        Ω ∈ PDiagDomain(1)
        σ ∈ RealDomain(lower = 0)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @pre begin
        Vc = tvVc * exp(η[1])
        mlp_ = only ∘ mlp  # technical
    end
    @dynamics begin
        Central' = mlp_(Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)# x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

fpm3 = fit(
    ude_model_3,
    population,
    init_params(ude_model_3),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 150),
)

pred = predict(ude_model_3, population[1:12], coef(fpm3); obstimes = 0:0.1:10);
plotgrid(pred)

# ALL HAVE SAME STARTING VALUE BUT POSSIBLY DIFFERENT CURVES
ude_model_4 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
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
        mlp_ = only ∘ mlp  # technical
    end
    @dynamics begin
        Central' = mlp_(Cl, Vc, Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)# x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

# RANDOM EFFECTS IN DYNAMICS to discharge MLP from patient specific characteritics
ude_model_3 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
        # tvCl ∈ RealDomain(lower = 0)
        tvVc ∈ RealDomain(lower = 0)
        Ω ∈ PDiagDomain(1)
        σ ∈ RealDomain(lower = 0)
    end
    @random begin
        η ~ MvNormal(Ω)
        η_nn ~ MvNormal(Ω)
    end
    @pre begin
        Vc = tvVc * exp(η[1])
        mlp_ = only ∘ mlp  # technical
    end
    @dynamics begin
        Central' = mlp_(Central, η_nn)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)# x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

# ALL HAVE SAME STARTING VALUE BUT POSSIBLY DIFFERENT CURVES

# CL as unique random effect should work equally well as model_1
# naming it CL is "mean" beacuse it doesn't mean CL anymore, it means random effect
ude_model_4 = @model begin
    @param begin
        mlp ∈ MLP(2, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
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
        mlp_ = only ∘ mlp  # technical
    end
    @dynamics begin
        Central' = mlp_(Cl, Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)# x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end

fpm4 = fit(
    ude_model_4,
    population,
    init_params(ude_model_4),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 50),
)

pred = predict(ude_model_4, population[1:12], coef(fpm4); obstimes = 0:0.1:10);
plotgrid(pred)

# CALL IT ETA_nn instead of CL, which wasn't its meaning anyway
ude_model_5 = @model begin
    @param begin
        mlp ∈ MLP(2, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
        # tvCl ∈ RealDomain(lower = 0)
        tvVc ∈ RealDomain(lower = 0)
        ω_Vc ∈ RealDomain(1)
        σ ∈ RealDomain(lower = 0)
    end
    @random begin
        η_Vc ~ Normal(0, ω_Vc)
        η_nn ~ Normal(0, 1)
    end
    @pre begin
        η_ = η_nn
        Vc = tvVc * exp(η_Vc)
        mlp_ = only ∘ mlp  # technical
    end
    @dynamics begin
        Central' = mlp_(η_, Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)# x1000 to match dose (mg) and concentration (μg/L)
        dv ~ @. Normal(cp, σ)
    end
end
