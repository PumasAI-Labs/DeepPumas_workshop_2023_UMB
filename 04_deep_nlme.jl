using DeepPumas
using StableRNGs
using PumasUtilities
using CairoMakie
using JuliaFormatter


#
# TABLE OF CONTENTS
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
preds = predict(data_model, population[1:4], init_params(data_model));
plotgrid(preds)

# EXAMPLE WITH UDE HAVING ONLY CENTRAL AND NO RANDOM EFFECTS 
# THIS MEANS THAT THERE ISN'T ANY PERSONALIZATION AND PREDICTIONS
# ARE EXACTLY THE SAME FOR ALL PATIENTS
ude_model_2 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity); reg = L2(1.0))
        # tvCL ∈ RealDomain(lower = 0)
        tvVc ∈ RealDomain(lower = 0)
        # Ω ∈ PDiagDomain(2)
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

fmlp2 = only ∘ coef(fpm2).mlp
central_prime = [fmlp2(central) for central in 0:0.1:5]
lines(
    0:0.1:5,
    central_prime;
    axis = (xlabel = "Central", ylabel = "Central' = mlp(Central)"),
)

# EXAMPLE WITH UDE HAVING ONLY CENTRAL BUT ADDING RANDOM EFFECTS 

ude_model_3 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
        # tvCL ∈ RealDomain(lower = 0)
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
        cp := @. 1000 * (Central / Vc)# x1000 to match concentration (μg/L) to dose (mg)
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

pred = predict(ude_model_3, population[1:4], coef(fpm3); obstimes = 0:0.1:10);
plotgrid(pred)

# ALL HAVE SAME STARTING VALUE BUT POSSIBLY DIFFERENT CURVES

# CL as unique random effect should work equally well as model_1
# naming it CL is "mean" beacuse it doesn't mean CL anymore, it means random effect
ude_model_4 = @model begin
    @param begin
        mlp ∈ MLP(2, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
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
        mlp_ = only ∘ mlp  # technical
    end
    @dynamics begin
        Central' = mlp_(CL, Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)# x1000 to match concentration (μg/L) to dose (mg)
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
        # tvCL ∈ RealDomain(lower = 0)
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
        cp := @. 1000 * (Central / Vc)# x1000 to match concentration (μg/L) to dose (mg)
        dv ~ @. Normal(cp, σ)
    end
end

# ABOVE AND BELOW ARE THE SAME I THINK

# RANDOM EFFECTS IN DYNAMICS to discharge MLP from patient specific characteritics
ude_model_3 = @model begin
    @param begin
        mlp ∈ MLP(1, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
        # tvCL ∈ RealDomain(lower = 0)
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
        cp := @. 1000 * (Central / Vc)# x1000 to match concentration (μg/L) to dose (mg)
        dv ~ @. Normal(cp, σ)
    end
end




""" 
Model with a universal differential equation and randon effects.
"""
# ALL HAVE SAME STARTING VALUE BUT POSSIBLY DIFFERENT CURVES
# discuss how CL and Vc are actually interchangeable if using Central' = mlp_(CL, Vc, Central)[1]

ude_model = @model begin
    @param begin
        mlp ∈ MLP(3, 4, 4, (1, identity); reg = L2(1.0))  # DEFAULT NON-LIN IN HIDDEN LAYERS IS TANH, TRY OUT HERE
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
        Central' = mlp_(CL, Vc, Central)
    end
    @derived begin
        cp := @. 1000 * (Central / Vc)
        dv ~ @. Normal(cp, σ)
    end
end

fpm = fit(
    ude_model,
    population,
    init_params(ude_model),
    MAP(FOCE());
    diffeq_options = (; alg = Rodas5P()),
    optim_options = (; iterations = 150),
)

pred = predict(ude_model, population[1:4], coef(fpm));
plotgrid(pred[1:4])

#  DISCUSS THIS Model
# IDENTIFIABILITY ISSUES
# EXAMINING THE MLP ON ITS OWN, FOR EXAMPLE PLOTTING
fmlp = only ∘ coef(fpm).mlp
central_prime = [fmlp(1.0, 1.0, central) for central in 0:0.1:5]
lines(
    0:0.1:5,
    central_prime;
    axis = (xlabel = "Central", ylabel = "Central' = mlp(CL=1, Vc=1, Central)"),
)

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


