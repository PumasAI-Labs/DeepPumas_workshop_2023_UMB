using DeepPumas
using PumasPlots
using PumasPlots.CairoMakie

#TODO Add units to my comments and to plots

"""
Helper Pumas model to generate synthetic data.
It assumes one compartment and oral dosing. #TODO Name of dynamics?
"""
data_model = @model begin
    @param begin
        tvImax ∈ RealDomain(; lower = 0.0, init = 1.1)  # typical value of maximum inhibition
        tvIC50 ∈ RealDomain(; lower = 0.0, init = 0.8)  # typical value of concentration for half-way inhibition
        tvKa ∈ RealDomain(; lower = 0.0)                # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0, init = 0.05)      # residual error
    end
    @pre begin
        Imax = tvImax                                   # per subject value = typical value,
        IC50 = tvIC50                                   # that is, no subject deviations, or,
        Ka = tvKa                                       # in other words, no random effects
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - Imax * Central / (IC50 + Central)
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

# Simulate two one-subject populations with same 
# `true_parameters` but different dosage regimens

true_parameters = (; tvImax = 1.1, tvIC50 = 0.8, tvKa = 1.0, σ = 0.1)

simobs_medium_dose = simobs(
    data_model,
    Subject(; events = DosageRegimen(5.0)),
    true_parameters;
    obstimes = 0:2:10,
)
pop_medium_dose = [Subject(simobs_medium_dose)]
plotgrid(pop_medium_dose)

simobs_large_dose = simobs(
    data_model,
    Subject(; events = DosageRegimen(15.0)),
    true_parameters;
    obstimes = 0:1:20,
)
pop_large_dose = [Subject(simobs_large_dose)]
plotgrid!(pop_large_dose)


# A SciML model where the full dynamics of central is unknown
ude_model = @model begin
    @param begin
        mlp ∈ MLP(2, 6, 6, (1, identity); reg = L2(0.5))    # neural network with 2 inputs and 1 output
        tvKa ∈ RealDomain(; lower = 0.0)                    # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)                       # residual error
    end
    @pre begin
        mlp_ = mlp
        Ka = tvKa
    end
    @dynamics begin
        Depot' = -Ka * Depot                # `Depot` dynamics are known
        Central' = mlp_(Depot, Central)[1]  # `Central` dynamics are completely unknown
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

fpm1 = fit(ude_model, pop_medium_dose, init_params(ude_model), MAP(NaivePooled()))

# The interpolation performance, and usually extrapolation, tends to be fairly
# good. It depends a bit on the data (noise, sparsity, identifiability).
plotgrid(predict(fpm1; obstimes = 0:0.1:15))

# How does this preform under a different dose?
plotgrid(predict(ude_model, pop_large_dose, coef(fpm1); obstimes = 0:0.1:15))

# Let's try again but this time we encode some more knowledge into the model,
#leaving less for the ML to capture.

model2 = @model begin
    @param begin
        NN ∈ MLP(1, 4, 4, (1, identity, false); reg = L2(0.5))
        tvKa ∈ RealDomain(; lower = 0.0)
        σ ∈ RealDomain(; lower = 0.0)
    end
    @pre begin
        Ka = tvKa
        _NN = NN
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - _NN(Central)[1]
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end


fpm2 = fit(model2, pop_medium_dose, sample_params(model2), MAP(NaivePooled()))

plotgrid(predict(fpm2; obstimes = 0:0.1:15))

# Now, again with the dose we never trained on.
plotgrid(predict(model2, pop_large_dose, coef(fpm2); obstimes = 0:0.1:25))

# Perhaps I'm guilty of tweaking the data generation until the first example
# extrapolated poorly to new doses while the second one does well... But it
# still get's the point across that encoding more scientific knowledge into the
# model just makes it better.
