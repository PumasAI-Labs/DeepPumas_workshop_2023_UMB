using DeepPumas
using PumasPlots
using PumasPlots.CairoMakie
using StableRNGs

# 
# TABLE OF CONTENTS
# 
# 1. INTRODUCTION
#
# 1.1. Simulate subjects A and B with different dosage regimens
# 1.2. A dummy neural network for modeling dynamics
# 
# 2. IDENTIFICATION OF MODEL DYNAMICS USING NEURAL NETWORKS
#
# 2.1. Delegate the identification of dynamics to a neural network
# 2.2. Combine existing domain knowledge and a neural network
# 2.3. Extend the analysis to a population of multiple subjects
# 2.4. Analyse the effect of very sparse data on the predictions
#

# 
# 1. INTRODUCTION
#
# 1.1. Simulate subjects A and B with different dosage regimens
# 1.2. A dummy neural network for modeling dynamics
# 

"""
Helper Pumas model to generate synthetic data. It assumes 
one compartment non-linear elimination and oral dosing.
"""
data_model = @model begin
    @param begin
        tvImax ∈ RealDomain(; lower = 0.0)  # typical value of maximum inhibition
        tvIC50 ∈ RealDomain(; lower = 0.0)  # typical value of concentration for half-way inhibition
        tvKa ∈ RealDomain(; lower = 0.0)    # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)       # residual error
    end
    @pre begin
        Imax = tvImax                       # per subject value = typical value,
        IC50 = tvIC50                       # that is, no subject deviations, or,
        Ka = tvKa                           # in other words, no random effects
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - Imax * Central / (IC50 + Central)
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

true_parameters = (; tvImax = 1.1, tvIC50 = 0.8, tvKa = 1.0, σ = 0.1)

# 1.1. Simulate subjects A and B with different dosage regimens

sim_a = simobs(
    data_model,
    Subject(; events = DosageRegimen(5.0)),
    true_parameters;
    obstimes = 0:1:10,
)
plotgrid([Subject(sim_a)]; data = (; label = "Data (subject A)"))

sim_b = simobs(
    data_model,
    Subject(; events = DosageRegimen(10.0)),  # higher dose
    true_parameters;
    obstimes = 0:1:10,
)
plotgrid!([Subject(sim_b)]; data = (; label = "Data (subject B)"), color = :gray)

# 1.2. A dummy neural network for modeling dynamics

time_model = @model begin
    @param begin
        mlp ∈ MLP(1, 6, 6, (1, identity); reg=L2(0.5))
        σ ∈ RealDomain(; lower = 0.0)
    end
    @derived Outcome ~ @. Normal(only(mlp(t)), σ)
end

pop_a = read_pumas(DataFrame(sim_a); observations = [:Outcome], event_data = false)
fpm = fit(time_model, pop_a, init_params(time_model), MAP(NaivePooled()))
pred_a = predict(fpm; obstimes=0:0.1:10);
plotgrid!(pred_a; pred = (; label = "Pred (subject A)"), ipred = false)

pop_b = read_pumas(DataFrame(sim_b); observations = [:Outcome], event_data = false)
pred_b = predict(time_model, pop_b, coef(fpm); obstimes=0:0.1:10);
plotgrid!(pred_b, pred = (; label = "Pred (subject B)", color = :red), ipred = false)

# 
# 2. IDENTIFICATION OF MODEL DYNAMICS USING NEURAL NETWORKS
#
# 2.1. Delegate the identification of dynamics to a neural network
# 2.2. Combine existing domain knowledge and a neural network
# 2.3. Extend the analysis to a population of multiple subjects
# 2.4. Analyse the effect of very sparse data on the predictions
#

# 2.1. Delegate the identification of dynamics to a neural network

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

plotgrid([Subject(sim_a)]; data = (; label = "Data (subject A)"))
plotgrid!([Subject(sim_b)]; data = (; label = "Data (subject B)"), color = :gray)

fpm = fit(ude_model, [Subject(sim_a)], init_params(ude_model), MAP(NaivePooled()))
pred_a = predict(fpm; obstimes=0:0.1:10);
plotgrid!(pred_a; pred = (; label = "Pred (subject A)"), ipred = false)

pred_b = predict(ude_model, [Subject(sim_b)], coef(fpm); obstimes=0:0.1:10);
plotgrid!(pred_b, pred = (; label = "Pred (subject B)", color = :red), ipred = false)

# 2.2. Combine existing domain knowledge and a neural network

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

fpm = fit(
    ude_model_knowledge,
    [Subject(sim_a)],
    init_params(ude_model_knowledge),
    MAP(NaivePooled()),
)

plotgrid([Subject(sim_a)]; data = (; label = "Data (subject A)"))
plotgrid!([Subject(sim_b)]; data = (; label = "Data (subject B)"), color = :gray)

pred_a = predict(fpm; obstimes=0:0.1:10);
plotgrid!(pred_a; pred = (; label = "Pred (subject A)"), ipred = false)

pred_b = predict(ude_model_knowledge, [Subject(sim_b)], coef(fpm); obstimes=0:0.1:10);
plotgrid!(pred_b, pred = (; label = "Pred (subject B)", color = :red), ipred = false)

# 2.3. Extend the analysis to a population of multiple, heterogeneous, subjects
#

data_model_heterogeneous = @model begin
    @param begin
        tvImax ∈ RealDomain(; lower = 0.0)  # typical value of maximum inhibition
        tvIC50 ∈ RealDomain(; lower = 0.0)  # typical value of concentration for half-way inhibition
        tvKa ∈ RealDomain(; lower = 0.0)    # typical value of absorption rate constant
        σ ∈ RealDomain(; lower = 0.0)       # residual error
    end
    @random η ~ MvNormal(Diagonal([0.1, 0.1, 0.1]))
    @pre begin
        Imax = tvImax * exp(η[1])
        IC50 = tvIC50 * exp(η[2])
        Ka = tvKa * exp(η[2]) 
    end
    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - Imax * Central / (IC50 + Central)
    end
    @derived begin
        Outcome ~ @. Normal(Central, σ)
    end
end

population = synthetic_data(data_model_heterogeneous, DosageRegimen(5.0), true_parameters; obstimes=0:1:10, nsubj=25, rng=StableRNG(1))
plotgrid(population)

fpm = fit(
    ude_model_knowledge,
    population,
    init_params(ude_model_knowledge),
    MAP(NaivePooled()),
)

pred = predict(fpm; obstimes=0:0.1:10);
plotgrid(pred)

begin
    f = nothing
    for (i, p) in enumerate(pred)
        if i == 1
            f = plotgrid([p]; ipred = false, title = "")
        else
            plotgrid!(
                [p];
                data = (; color = Cycled(i)),
                ipred = false,
                title = "",
                add_legend = false,
            )
        end
    end
    f
end

# 2.4. Analyse the effect of very sparse data on the predictions

sims_sparse = [
    simobs(
        data_model_heterogeneous,
        Subject(; events = DosageRegimen(5.0), id = i),
        true_parameters;
        obstimes = 11 .* sort!(rand(2)),
    ) for i = 1:25
]
population_sparse = Subject.(sims_sparse)

fpm = fit(
    ude_model_knowledge,
    population_sparse,
    init_params(ude_model_knowledge),
    MAP(NaivePooled()),
)

pred = predict(fpm; obstimes = 0:0.01:10);
plotgrid(pred)

begin
    f = nothing
    for (i, p) in enumerate(pred)
        if i == 1
            f = plotgrid([p]; ipred = false, title = "")
        else
            plotgrid!(
                [p];
                data = (; color = Cycled(i)),
                ipred = false,
                title = "",
                add_legend = false,
            )
        end
    end
    f
end
