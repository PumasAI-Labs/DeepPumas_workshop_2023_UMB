using DeepPumas
using PumasPlots
using PumasPlots.CairoMakie

datamodel = @model begin
  @param begin
    tvImax ∈ RealDomain(; lower=0.)
    tvIC50 ∈ RealDomain(; lower=0.)
    tvKa ∈ RealDomain(; lower=0.)
    σ ∈ RealDomain(; lower=0.)
  end
  @pre begin
    Ka = tvKa 
    Imax = tvImax
    IC50 = tvIC50 
  end
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - Imax * Central / (IC50 + Central)
  end
  @derived begin
    # Essentially an additive noise model but constrained to positive observations.
    Outcome ~ @. Gamma((Central + 1e-10)^2/σ^2, σ^2 / (Central + 1e-10))
  end
end

p_true = (; tvImax = 1.1, tvIC50=0.8, tvKa=1., σ = 0.2)

# A single PK time course.
sim = simobs(datamodel, Subject(; events=DosageRegimen(5.)), p_true; obstimes=range(0, stop=10, length=10))
training_data = [Subject(sim)]

# Plot the training data.
# For more plotting options, enter ?plotgrid in the REPL.
plotgrid(training_data)

# Synthetic data for testing the models on a dose they've not been trained on.
sim_large_dose = simobs(datamodel, Subject(; events=DosageRegimen(15.)), p_true; obstimes=0:1:20)
test_data_large_dose = [Subject(sim_large_dose)]

# Synthetic data for testing the models on a dose they've not been trained on.
sim_multi_dose = simobs(datamodel, Subject(; events=DosageRegimen(5.; ii=5, addl=2)), p_true; obstimes=0:1:20)
test_data_multi_dose = [Subject(sim_multi_dose)]


ml_model = @model begin
  @param begin
    NN ∈ MLP(1, 6, 6, (1, identity); reg=L2(0.1)) # much more regularization and you'll just get a straight line
    σ ∈ RealDomain(; lower=0.)
  end
  @pre X = NN(t)[1]
  @derived Outcome ~ @. Normal(X, σ)
end

# This model does not have a mechanism for handling doses so we can't really use that
# information and we need to make a population without doses.
training_data_no_dose = read_pumas(DataFrame(sim); observations=[:Outcome], event_data=false)

fpm_ml = fit(ml_model, training_data_no_dose, sample_params(ml_model), MAP(NaivePooled()))
plotgrid(predict(fpm_ml; obstimes=0:0.1:15))
# This model hits the data points just fine but it would have trouble doing extrapolation
# and it does not respect any of our scientific knowledge of PK curves. Furthermore, we
# don't even have a mechanism for training on one dosage regimen and predicting outcomes
# from another.

############################################################################################
# Scientific machine learning (SciML) using universal differential equations (UDEs).
############################################################################################
# Let's instead create a SciML model where we encode knowledge about the PK being oral but
# where we say that the dynamics of central is entirely unknown
sciml_model_1 = @model begin
  @param begin
    NN ∈ MLP(2, 6, 6, (1, identity); reg=L2(1.0))
    tvKa ∈ RealDomain(; lower=0.)
    σ ∈ RealDomain(; lower=0.)
  end
  @pre begin
    Ka = tvKa 
    _NN = NN
  end
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = _NN(Depot, Central)[1]
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end

# Train on a single dose with just a few observations.
fpm1 = fit(sciml_model_1, training_data, sample_params(sciml_model_1), MAP(NaivePooled()))

# The interpolation, and usually extrapolation, performance of SciML models tend to be
# fairly good. It depends a bit on the data (noise, sparsity, identifiability) and might
# change if you re-generate the data.
plotgrid(predict(fpm1; obstimes=0:0.1:20))

# How does this perform under different dosing regimens?
plotgrid(predict(sciml_model_1, test_data_multi_dose, coef(fpm1); obstimes=0:0.1:20))

plotgrid(predict(sciml_model_1, test_data_large_dose, coef(fpm1); obstimes=0:0.1:15))


# There's a bit of randomness involved (mostly from the data generation), but I'm guessing
# that you're seeing how the model works pretty well on the dose it was trained on and even
# extrapoletes well to mutiple doses. Already we're seeing huge utility over a more
# traditional ML approach. However, the model does not extrapolate very well with doses that
# are substantially higher than trained on so there's still room for improvement.


# Let's try again but this time we encode some more knowledge into the model, leaving less
# for the ML to capture.
sciml_model_2 = @model begin
  @param begin
    NN ∈ MLP(1, 4, 4, (1, identity, false); reg=L2(0.5))
    tvKa ∈ RealDomain(; lower=0.)
    σ ∈ RealDomain(; lower=0.)
  end
  @pre begin
    Ka = tvKa 
    _NN = NN
  end
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - _NN(Central)[1]
  end
  @derived begin
    Outcome ~ @. Normal(Central, σ)
  end
end


fpm2 = fit(sciml_model_2, training_data, sample_params(sciml_model_2), MAP(NaivePooled()))

plotgrid(predict(fpm2; obstimes=0:0.1:15))

# Did the incorporation of further scientific knowledge help the model extrapolate better?
plotgrid(predict(sciml_model_2, test_data_multi_dose, coef(fpm2); obstimes=0:0.1:15))
plotgrid(predict(sciml_model_2, test_data_large_dose, coef(fpm2); obstimes=0:0.1:25))

# Perhaps I'm guilty of tweaking the data generation until sciml_model_1 extrapolated poorly
# to high doses while sciml_model_2 does well... But it still get's the point across that
# encoding more scientific knowledge into the model just makes it better.

#=
Bonus exercise:
Tweak the generation of training data and see what might help or hurt the models' ability to
extrapolate well.  There are three main considerations to probe:
- Noisiness - tweak σ in p_true
- Number of observations - tweak obstimes in the simobs that generates the training data.
- IC50 identifiability - what happens if the training data never had a dose exceeding IC50? Can the model then extrapolate to doses well above IC50?
=#


# But, wait... This is not the kind of data we typically have... We tend to have many
# patients, each with their own longitudinal measurments and where each patient is a bit different.


datamodel_pop = @model begin
  @param begin
    tvImax ∈ RealDomain(; lower=0.)
    tvIC50 ∈ RealDomain(; lower=0.)
    tvKa ∈ RealDomain(; lower=0.)
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ MvNormal(Diagonal([0.2, 0.2, 0.2]))
  @pre begin
    Ka = tvKa * exp(η[1])
    Imax = tvImax* exp(η[2])
    IC50 = tvIC50 * exp(η[2])
  end
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - Imax * Central / (IC50 + Central)
  end
  @derived begin
    Outcome ~ @. Gamma((Central + 1e-10)^2/σ^2, σ^2 / (Central + 1e-10))
  end
end

sims = [simobs(datamodel_pop, Subject(; events=DosageRegimen(5.)), p_true; obstimes=range(0, stop=10, length=6)) for _ in 1:12]
training_population = Subject.(sims)
plotgrid(training_population)

fpm_pop_1 = fit(sciml_model_2, training_population, init_params(sciml_model_2), MAP(NaivePooled()))

pred = predict(fpm_pop_1; obstimes = 0:0.01:15)
plotgrid(pred)

# Let's plot all of them ontop of oneanother
begin
  plt = plotgrid([pred[1]])
  for i in 2:length(pred)
    plotgrid!([pred[i]]; data=(; color=Cycled(i)), add_legend=false)
  end
  plt
end

# Here, the SciML approach has discovered a model that identifies the average longitudinal
# behaviour of our heterogeneous data. We have not been able to predict or account for
# inter-patient variability, but we've been able to use data from multiple individuals.

# This could come in handy if you could, for example only sample a few datapoints per
# subject but where you assume that they all share some trend over time. To demonstrate,
# we're here randomly picking only two times for the PK observations for each synthetic
# subject and then we fit one of the sciml models on the whole population.

sims2 = [simobs(datamodel_pop, Subject(; events=DosageRegimen(5.)), p_true; obstimes=10 .* rand(2)) for _ in 1:25]
training_population2 = Subject.(sims2)
fpm_pop_2 = fit(sciml_model_2, training_population2, init_params(sciml_model_2), MAP(NaivePooled()))

pred2 = predict(fpm_pop_2; obstimes = 0:0.01:15)
plotgrid(pred2)

begin
  plt = plotgrid([pred2[1]])
  for i in 2:length(pred)
    plotgrid!([pred2[i]]; data=(; color=Cycled(i)), add_legend=false)
  end
  plt
end

# The methodology used thus far has not really enabled us to account for heterogeneous data
# and so the predictions are the same for all patients. To account for heterogeniety, we
# will need to take yet another step in this ongoing saga and explore DeepNLME.


data_multi_subject_multi_dose = Subject.(simobs(datamodel_pop, Subject(; events=DosageRegimen(5.; ii=5, addl=2)), p_true; obstimes=0:1:20))