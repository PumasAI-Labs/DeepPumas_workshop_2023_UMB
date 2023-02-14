using DeepPumas
using PumasPlots
using PumasPlots.CairoMakie

datamodel = @model begin
  @param begin
    tvImax ∈ RealDomain(; lower=0., init=1.1)
    tvIC50 ∈ RealDomain(; lower=0., init=0.8)
    tvKa ∈ RealDomain(; lower=0.)
    σ ∈ RealDomain(; lower=0., init=0.05)
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
    Outcome ~ @. Normal(Central, σ)
  end
end

p_true = (; tvImax = 1.1, tvIC50=0.8, tvKa=1., σ = 0.1)
sim = simobs(datamodel, Subject(; events=DosageRegimen(5.)), p_true; obstimes=range(0, stop=10, length=6))
pop = [Subject(sim)]

sim_large_dose = simobs(datamodel, Subject(; events=DosageRegimen(15.)), p_true; obstimes=0:1:20)
pop_large_dose = [Subject(sim_large_dose)]

plotgrid(pop)

# A SciML model where the full dynamics of central is unknown
model1 = @model begin
  @param begin
    NN ∈ MLP(2, 6, 6, (1, identity); reg=L2(0.5))
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

fpm1 = fit(model1, pop, init_params(model1), MAP(NaivePooled()))

# The interpolation performance, and usually extrapolation, tends to be fairly
# good. It depends a bit on the data (noise, sparsity, identifiability).
plotgrid(predict(fpm1; obstimes=0:0.1:15))

# How does this preform under a different dose?
plotgrid(predict(model1, pop_large_dose, coef(fpm1); obstimes=0:0.1:15))

# Let's try again but this time we encode some more knowledge into the model,
#leaving less for the ML to capture.

model2 = @model begin
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


fpm2 = fit(model2, pop, sample_params(model2), MAP(NaivePooled()))

plotgrid(predict(fpm2; obstimes=0:0.1:15))

# Now, again with the dose we never trained on.
plotgrid(predict(model2, pop_large_dose, coef(fpm2); obstimes=0:0.1:25))

# Perhaps I'm guilty of tweaking the data generation until the first example
# extrapolated poorly to new doses while the second one does well... But it
# still get's the point across that encoding more scientific knowledge into the
# model just makes it better.