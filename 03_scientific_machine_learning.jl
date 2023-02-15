using DeepPumas
using PumasPlots
using PumasPlots.CairoMakie

#TODO Add units to my comments and to plots

# 
# TABLE OF CONTENTS
# 
# 1. IDENTIFICATION OF MODEL DYNAMICS
# 1.1. Sample data based on a known model
# 1.2. Delegate the identification of dynamics to a neural network
# 1.3. Exercise: Assess the quality of predictions on higher doses
# 1.4. Combine existing domain knowledge and a neural network
# 1.5. Exercise: Revisit exercise 1.3 with the combined model
#

# 1.1 Sample data based on a known model

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

# sample a population of one subject only!

true_parameters = (; tvImax = 1.1, tvIC50 = 0.8, tvKa = 1.0, σ = 0.1)

sim = simobs(
    data_model,
    Subject(; events = DosageRegimen(5.0)),
    true_parameters;
    obstimes = 0:2:10,
)
pop = [Subject(sim)]

plotgrid(pop)

# 1.2. Delegate the identification of dynamics to a neural network

ude_model = @model begin
    @param begin
        mlp ∈ MLP(2, 6, 6, (1, identity); reg = L2(0.5))    # neural network with 2 inputs and 1 output
        tvKa ∈ RealDomain(; lower = 0.0)                    # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)                       # residual error
    end
    @pre begin
        mlp_ = only ∘ mlp
        Ka = tvKa
    end
    @dynamics begin
        Depot' = -Ka * Depot                                # known
        Central' = mlp_(Depot, Central)                     # left as function of `Depot` and `Central`
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

fpm = fit(ude_model, pop, init_params(ude_model), MAP(NaivePooled()))

# The interpolation performance, and usually extrapolation, tends to be fairly
# good. It depends a bit on the data (noise, sparsity, identifiability).
plotgrid(predict(fpm; obstimes = 0:0.1:15))

# 1.3. Exercise: Assess the quality of predictions on higher doses

solution_ex13 = begin

    # sample another population with the same `true_parameters` but higher doses
    simobs_higher_dose = simobs(
        data_model,
        Subject(; events = DosageRegimen(15.0)),    # compared to 5.0 mg above
        true_parameters;                            # same as above
        obstimes = 0:1:20,
    )
    pop_higher_dose = [Subject(simobs_higher_dose)]

    plotgrid!(pop_higher_dose)

    # predictions of `ude_model` on higher dose
    plotgrid(predict(ude_model, pop_higher_dose, coef(fpm); obstimes = 0:0.1:15))

end

# 1.4. Combine existing domain knowledge and a neural network

ude_model_knowledge = @model begin
    @param begin
        mlp ∈ MLP(1, 6, 6, (1, identity); reg = L2(0.5))    # neural network with 2 inputs and 1 output
        tvKa ∈ RealDomain(; lower = 0.0)                    # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)                       # residual error
    end
    @pre begin
        mlp_ = only ∘ mlp
        Ka = tvKa
    end
    @dynamics begin
        Depot' = -Ka * Depot                                # known
        Central' = Ka * Depot - mlp_(Central)               # knowledge of conservation added
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

fpm_knowledge =
    fit(ude_model_knowledge, pop, init_params(ude_model_knowledge), MAP(NaivePooled()))

plotgrid(predict(fpm_knowledge; obstimes = 0:0.1:15))  # prediction on `pop`

# 1.5. Exercise: Revisit exercise 1.3 with the combined model

solution_ex15 = begin
    plotgrid(
        predict(
            ude_model_knowledge,
            pop_higher_dose,
            coef(fpm_knowledge);
            obstimes = 0:0.1:25,
        ),
    )
end
