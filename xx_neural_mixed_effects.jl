using DeepPumas
using CairoMakie
using PumasPlots

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
    IC50 = tvIC50 * exp(η[3])
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
testpop = pop[end-5:end]
trainpop_no_dose = read_pumas(DataFrame(trainpop); observations=[:PK], event_data=false)
testpop_no_dose = read_pumas(DataFrame(testpop); observations=[:PK], event_data=false)

plotgrid(trainpop[1:12])

############################################################################################
# A benchmark from our previous excercises - a traditional machine learning model.
############################################################################################

model_ml = @model begin
  @param begin
    NN ∈ MLP(1, 6, 6, (1, identity); reg=L2(1.0)) 
    σ ∈ RealDomain(; lower=0.)
  end
  @pre X = NN(t)[1]
  @derived PK ~ @. Normal(X, σ)
end

fpm_ml = fit(model_ml, trainpop_no_dose, sample_params(model_ml), MAP(NaivePooled()))
pred_ml = predict(model_ml, testpop_no_dose, coef(fpm_ml); obstimes=0:0.1:10);
plotgrid(pred_ml)

# One curve for all. Here, the use of multiple patients is providing a wealth of data for
# the training and we don't overfit as much as in previous examples.
# Now, let's do DeepNLME where we have a random effect as input to the NN.
     
############################################################################################
# Model 1 - additive η
############################################################################################
model1 = @model begin
  @param begin
    NN ∈ MLP(1, 6, 6, (1, identity); reg=L2(1.0))
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ Normal(0., 1.)
  @pre X = NN(t)[1] + η
  @derived PK ~ @. Normal(X, σ)
end

fpm1 = fit(model1, trainpop_no_dose, sample_params(model1), MAP(FOCE()))
pred1 = predict(model1, testpop_no_dose, coef(fpm1); obstimes=0:0.1:10)

## What can this random effect do?
plotgrid(pred1)

############################################################################################
# Model 2 - exp(η)
############################################################################################
model2 = @model begin
  @param begin
    NN ∈ MLP(1, 6, 6, (1, identity); reg=L2(1.0))
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ Normal(0., 1.)
  @pre X = NN(t)[1] * exp(η)
  @derived PK ~ @. Normal(X, σ)
end

fpm2 = fit(model2, trainpop_no_dose, sample_params(model2), MAP(FOCE()))
pred2 = predict(model2, testpop_no_dose, coef(fpm2); obstimes=0:0.1:10)

## What can the random effect do?
plotgrid(pred2)


############################################################################################
# Model 3 - random effect as input to the NN
############################################################################################
model3 = @model begin
  @param begin
    NN ∈ MLP(2, 6, 6, (1, identity); reg=L2(1.0))
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ Normal(0., 1.0)
  @pre X = NN(t, η)[1]
  @derived PK ~ @. Normal(X, σ)
end

fpm3 = fit(model3, trainpop_no_dose, sample_params(model3), MAP(FOCE()))
pred3 = predict(model3, testpop_no_dose, coef(fpm3); obstimes=0:0.1:10)

## What can the random effect do?
plotgrid(pred3)


############################################################################################
# Model 4 - two random effects as input to the NN
############################################################################################
model4 = @model begin
  @param begin
    NN ∈ MLP(3, 6, 6, (1, identity); reg=L2(1.0))
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ MvNormal(Diagonal([1., 1.]))
  @pre X = NN(t, η)[1]
  @derived PK ~ @. Normal(X, σ)
end

fpm4 = fit(model4, trainpop_no_dose, sample_params(model4), MAP(FOCE()); optim_options=(; time_limit=120))
pred4 = predict(model4, testpop_no_dose, coef(fpm4); obstimes=0:0.1:10);
plotgrid(pred4)

ins = inspect(fpm4)
goodness_of_fit(ins)