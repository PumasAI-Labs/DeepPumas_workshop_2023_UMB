using DeepPumas
using PumasPlots
using PumasPlots.CairoMakie
using DeepPumas.SimpleChains: softplus
# using Pumas
using Serialization
using Random
# using DeepPumas.Pumas.Latexify
using Revise
Makie.set_theme!(DeepPumas.plottheme())

# Define the mean such that `predict` outputs probability of surviving until time t.
Pumas.Statistics.mean(t::TimeToEvent) = exp(-t.Λ) 
Pumas._unwrap(d::TimeToEvent) = d

## Hack to circumvent a bug triggered in _lpdf
Base.eltype(::TimeToEvent{T}) where T = T

include("utils/TSI_OS_data_generation.jl")

ntrain=200
## Observation times for plot generation. Dense sampling -> smooth plots.
obstimes=0:0.0025:1
title = (s, i) -> "ID:$(s.id), Dose: $(s.events[1].amt < 0.1 ? "None" : "$(s.events[1].amt)mg")"


datamodel = @model begin
  @param begin
    λ₁ ∈ RealDomain(; lower=0, init=0.1)
    β ∈ RealDomain(; init=0.001)
    tvCL ∈ RealDomain(; lower=0.0, init=30.0)
    tvVc ∈ RealDomain(; lower=0.0)
    tvPR ∈ RealDomain(; lower=0.0, init=11.0)
    tvSR ∈ RealDomain(; lower=0.0, init=20.0)
    M_BASE ∈ RealDomain(; lower=0.0, init=12.1)
    Ω ∈ PDiagDomain(; init=sqrt.([0.1, 0.2, 0.1, 0.01, 0.1, 0.1]))
    CV ∈ RealDomain(; lower=0.0, init=0.1)
    tvEC50s ∈ RealDomain(; lower=0.0, init=2.0)
    tvEC50g ∈ RealDomain(; lower=0.0, init=2.0)
  end
  @covariates c1 c2 c3 c4 c5 c6
  @random η ~ MvNormal(Ω)
  @pre begin
    BASE = M_BASE * exp(η[1] + 0.1 * c1)
    PR = tvPR * exp(η[2] + 0.4 * c2)
    SR = tvSR * exp(η[3] + 0.2 * c3)
    _λ₀ = λ₁ * exp(η[4] + 0.001 * c4)
    Vc = tvVc * exp(η[5] + 0.1 * c5)
    CL = tvCL
    EC50shrink = tvEC50s
    EC50grow = tvEC50g
    G = 15.0 * exp(η[6] + 0.2(c5 + c6))
  end
  @init TSᵣ = BASE
  @vars begin
    "Total tumor size"
    TS = TSᵣ + TSᵤ
    "Hazard as a nonlinear function of tumor size"
    λ = 1.5 * (_λ₀ + (TS / 10)^2)
    cp = Central / Vc
  end
  @dynamics begin
    "Drug"
    Central' = -CL / Vc * Central
    "Drug-responsive tumor size"
    TSᵣ' = G * EC50grow / (EC50grow + cp) -
           SR * cp / (EC50shrink + cp) * TSᵣ
    "Drug-unresponsive tumor size"
    TSᵤ' = PR
    "Risk is the integal of hazard"
    Λ' = λ
  end
  @observed drug = @. Central / Vc
  @derived begin
    "Tumor size observations"
    TSO ~ @. Gamma(1/CV^2, CV^2 * TS) # CV is the coefficient of variation. The latent tumor size variable it the mean of the distribution.
    Survival ~ @. TimeToEvent(λ, Λ)
  end
end


rng = StableRNGs.StableRNG(111)
## Make different dosing regimens for each patient
drs = [DosageRegimen(Float64(rand(rng, 1:3)), addl=rand(rng, 1:10), ii=0.05 + 0.1 * rand(rng)) for _ in 1:800]
no_treatment = fill(DosageRegimen(1e-10), 200)
dr = shuffle(StableRNGs.StableRNG(10), vcat(drs, no_treatment))

# dr = [DosageRegimen(Float64(rand(rng, 1:3)), addl=25, ii=1/24) for _ in 1:1000]

full_pop, datasim = generate_os_data(datamodel, dr; nsubj=length(dr), obstimes=1e-10 .+ (0:12.0) ./ 12, rng=StableRNGs.StableRNG(123), dvname=:Survival)

data_ηs = map(datasim) do s
  s.randeffs
end

plotgrid(
  [predict(datamodel, full_pop[i], init_params(datamodel), data_ηs[i]) for i in 1:24]; 
  title
)

plotgrid(
  [simobs(datamodel, full_pop[i], init_params(datamodel), data_ηs[i]; simulate_error=false, obstimes) for i in 1:24]; 
  title,
  observation=:drug,
  markersize=0
)

model_ts = @model begin
    @param begin
      NNᵣ ∈ MLP(5, 7, 7, (1, identity); reg=L2(0.01; output=false))
      NNᵤ ∈ MLP(2, 4, 4, (1, identity); reg=L2(0.5))
      tvCL ∈ RealDomain(; lower=0.)
      tvVc ∈ RealDomain(; lower=0.)
      σ ∈ RealDomain(; lower=0., init=0.2)
      M_BASE ∈ RealDomain(; lower=0., init=10.)
      ω_BASE ∈ RealDomain(; lower=0., init=0.5)
      ωVc ∈ RealDomain(; lower=0.)
    end
    @random begin
      ηVc ~ Normal(0., ωVc)
      η ~ MvNormal(Float64.(I(3)))
      η_BASE ~ Normal(0., ω_BASE)
    end
    @pre begin
      CL = tvCL 
      Vc = tvVc * exp(ηVc)
      TS₀ = M_BASE * exp(η_BASE)

      # Just pass things on so that we can use them in the @dynamics
      _η = η
      _NNᵣ = NNᵣ
      _NNᵤ = NNᵤ
    end
    @init begin
      TSᵣ = TS₀
    end
    @vars begin
      "Total tumour size"
      TS = TSᵣ + TSᵤ
      cp = Central / Vc
    end
    @dynamics begin
      Central' = - CL/Vc * Central
      "Drug-responsive tumour size"
      TSᵣ' = _NNᵣ(cp, TSᵣ, _η[1], _η[2], _η[3])[1]
      "Drug-inresponsive tumour size"
      TSᵤ' = _NNᵤ(TSᵤ, _η[1])[1]
    end
    @derived begin
      TSO = @. Normal(TS, σ)
    end
  end

#= 
This model only models TSO but our data has multiple different quantities that are observed.
To reduce the risk of user-error, Pumas' `fit` will throw an error if your data has more
observations than your model is modelling. So, here, we make a copy of the data where we
filter out all observations except `TSO`.

Info: if you're unfamiliar with the "do" syntax below, you can find more info at 
https://docs.julialang.org/en/v1/manual/functions/#Do-Block-Syntax-for-Function-Arguments
=#
pop_TS = map(full_pop) do s
  Subject(s.id, NamedTuple{(:TSO,)}(s.observations), s.covariates, s.events, s.time)
end

trainpop_TS = @view pop_TS[1:ntrain]
testpop_TS = @view pop_TS[ntrain+1:end]

#=
Fitting with the entire population can be a bit slow so before we trigger such a fit, we fit
on small random subsets of the population. That should identify how the state variables
relate to oneanother. A subsequent full population fit should then figure out the best
parameterization of the NNs in order to produce the best ipreds.
=#

resample_params = true
nrepeats = 10
for _ in 1:nrepeats
  global fpm = fit(
    model_ts,
    sample(trainpop_TS, 30; replace=false),
    resample_params ? sample_params(model_ts) : coef(fpm),
    MAP(FOCE());
    optim_options = (; time_limit=2*60, iterations=20),
    diffeq_options = (; alg = Rodas5P()),
    checkidentification=false
  )
  display(plotgrid(predict(model_ts, testpop_TS[1:24], coef(fpm))))
  resample_params = false
end
plotgrid(predict(model_ts, testpop_TS[1:24], coef(fpm); obstimes))


fpm = fit(
  model_ts,
  trainpop_TS,
  sample_params(model_ts),
  MAP(FOCE());
  optim_options = (; time_limit=15*60),
  diffeq_options = (; alg = Rodas5P()),
)
serialize(@__DIR__() * "/fpm.jls", fpm)
plotgrid(predict(model_ts, testpop_TS[1:24], coef(fpm); obstimes))

# Letting this fit finish would likely yield better results, but that would take a little
# too long for a workshop. We'll just have to live with imperfection and remember that a
# more compute could have made things better.

############################################################################################
# DeepPumas covariate modelling. Discover how baseline covariates predict individial parameters.
############################################################################################

#=
We could let an NN process the covariates right in the @pre block but doing so is typically rater slow and a bit prone to get stuck in local optima. A good approach has been to first fit the NN externally, mapping from the covariates to the EBEs (actually, scaled and transformed variants thereof). This typically works nicely, but the quality is degraded by subjects who have poorly informed EBEs. We know that patients with too few observations will have poorly identified ηs, so let's remove them during this stage of the fitting.

Later, we can fit the covariate-processing NN jointly with the NLME model and then we don't
have these problems. This initial stage of fitting is both good for getting a (very) good
initial parameter guess for the full fit, and it could sometimes be used on its own if the
full fit is not supported or if it's computationally unfeasible. 
=#

filtered_trainpop_TS = filter(pop_TS) do s
  length(s.time) > 5
end

target = preprocess(model_ts, filtered_trainpop_TS, coef(fpm), LaplaceI())

nn = MLP(
  numinputs(target), # number of inputs
  7, # nodes in the first hidden layer
  7, # Nodes in the second hidden layer
  (numoutputs(target), identity);  
  reg=L1(0.7, output=false)
)
ho = hyperopt(nn, target)
model_ts_aug = augment(fpm, ho)

plotgrid(predict(model_ts, testpop_TS[1:25], coef(fpm); obstimes))
plotgrid!(predict(model_ts_aug, testpop_TS[1:25], init_params(model_ts_aug); obstimes); ipred=false, pred=(; color=CairoMakie.Cycled(3), label="Covariate model"))

#=
Fit dynamics and covariate model jointly. 

This is good for two things. The simplest is that it tunes the Ωs. The other reason is that
it fits the connection between covariates and outcome logliklihood a bit more directly.
This is especially good when we might have a few unidentifiable EBEs to gum up the works
in preprocess.

=#
fpmd_aug = fit(
  model_ts_aug,
  trainpop_TS,
  init_params(model_ts_aug),
  MAP(FOCE()); 
  optim_options = (; time_limit=15*60),
  diffeq_options = (; alg = Rodas5P()),
)


pop_DV = map(full_pop) do s
  Subject(s.id, NamedTuple{(:DV,)}(s.observations), s.covariates, s.events, s.time)
end
pop_TS_DV = map(full_pop) do s
  Subject(s.id, NamedTuple{(:TSO,:DV,)}(s.observations), s.covariates, s.events, s.time)
end



#=

Questions to ponder / explore

Given the data that we're using, do we expect every interesting aspect of the NN-embedded model to be identifiable? 

Here, all our patients receive monthly dosing. Their only difference is in dosing amount, 1, 2, or 3 mg. So, what can we expect to discover when it comes to modelling how the drug affects the patients? What kind of data would have enabled a full identification of precisely how the drug affects the patients? Can you think of a clinically feasible dosing/data collection strategy that would have provided better data for the model fitting and that you can ethically defend? That's not only an excercise question - I'm intrested to hear views on this!


=#