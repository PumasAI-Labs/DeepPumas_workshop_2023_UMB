using DeepPumas
using PumasPlots
using PumasPlots.CairoMakie

#
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
    PK ~ @. Normal(Central, Central * σ)
  end
end
p_true = (; tvImax = 1.1, tvIC50=0.8, tvKa=1., σ = 0.02)


sims = [simobs(datamodel_pop, Subject(; events=DosageRegimen(5.), id=i), p_true; obstimes=0:0.3:10) for i in 1:100]
pop = Subject.(sims)
trainpop =  pop[1:80]
testpop = pop[length(trainpop) + 1:end]
trainpop_no_dose = read_pumas(DataFrame(trainpop); observations=[:PK], event_data=false)
testpop_no_dose = read_pumas(DataFrame(testpop); observations=[:PK], event_data=false)

plotgrid(trainpop[1:12])


# A benchmark from our previous excercises - a somewhat traditional 
ml_model = @model begin
  @param begin
    NN ∈ MLP(1, 6, 6, (1, identity); reg=L2(1.0)) 
    σ ∈ RealDomain(; lower=0.)
  end
  @pre X = NN(t)[1]
  @derived Outcome ~ @. Normal(X, σ)
end

fpm_ml = fit(ml_model, trainpop_no_dose, sample_params(ml_model), MAP(NaivePooled()))
pred_ml = predict(ml_model, testpop_no_dose, coef(fpm_ml); obstimes=0:0.1:10)
plotgrid(pred_ml)
# One curve for all. Here, the use of multiple patients is providing a wealth of data for
# the training and we don't overfit as much as in previous examples.

# Now, let's do DeepNLME where we have a random effect as input to the NN.
ml_nlme_model = @model begin
  @param begin
    NN ∈ MLP(2, 6, 6, (1, identity); reg=L2(1.0))
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ Normal(0., 1.)
  @pre X = NN(t, η)[1]
  @derived Outcome ~ @. Normal(X, σ)
end

fpm_ml_nlme = fit(ml_nlme_model, trainpop_no_dose, sample_params(ml_nlme_model), MAP(FOCE()))
pred_ml_nlme = predict(ml_nlme_model, testpop_no_dose, coef(fpm_ml_nlme); obstimes=0:0.1:10)
plotgrid(pred_ml_nlme)

# Here, we have discovered a function for the pk-curve over time which takes one parameter
# and this parameter can account for much but not all of the variability between patients.


#=
However, the patients have variability across more than one dimension, so one η is not enough!

So, let's add another random effect - making η a vector of two.
=# 

ml_nlme_model_2 = @model begin
  @param begin
    NN ∈ MLP(3, 6, 6, (1, identity); reg=L2(1.0))
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ MvNormal(Diagonal([1., 1.]))
  @pre X = NN(t, η)[1]
  @derived Outcome ~ @. Normal(X, σ)
end

fpm_ml_nlme_2 = fit(ml_nlme_model_2, trainpop_no_dose, sample_params(ml_nlme_model_2), MAP(FOCE()); optim_options=(; time_limit=60))
pred_ml_nlme_2 = predict(ml_nlme_model_2, testpop_no_dose, coef(fpm_ml_nlme_2); obstimes=0:0.1:10)
plotgrid(pred_ml_nlme_2)
