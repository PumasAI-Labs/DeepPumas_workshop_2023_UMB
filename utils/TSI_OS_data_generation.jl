using DeepPumas

get_doses(::Nothing, nsubj) = fill(nothing, nsubj)
get_doses(dr::Pumas.DosageRegimen, nsubj) = fill(dr, nsubj)
function get_doses(dr::Vector{<:Union{Pumas.DosageRegimen,Nothing}}, nsubj)
  length(dr) !== nsubj && throw(ArgumentError(
    "generate_os_data: a supplied vector of dosage regimens must fulfil `length(dose_vector) == nsubj`."
  ))
  dr
end

function generate_os_data(
  model,
  doses,
  param=init_params(model);
  obstimes=0:0.05:0.99,
  covariates::NamedTuple=(;),
  rng=DeepPumas.default_rng(),
  nsubj=1000,
  dvname=:DV
)
  _covsyms = model.syms.covariates
  _covdists =
    NamedTuple{Tuple(_covsyms)}(map(s -> get(covariates, s, Normal()), Tuple(_covsyms)))
  _pop = if doses === nothing
    [Subject(; id=i, covariates=DeepPumas._rand_unnested(rng, _covdists)) for i in 1:nsubj]
  else
    [Subject(; id=i, events, covariates=DeepPumas._rand_unnested(rng, _covdists)) for (i, events) in enumerate(get_doses(doses, nsubj))]
  end

  sim = simobs(model, _pop, param; obstimes)

  # Simobs does not simulate death from TimeToEvent. Loop though the
  # simulations and sample time-of-death from based on the subjects' cumulative
  # hazard.
  full_pop = map(enumerate(sim)) do (i, s)
    _r = Pumas.randexp()
    cs = Pumas.DataInterpolations.CubicSpline(getfield.(s.observations[dvname], :Λ), obstimes[1:end])
    if s.observations[dvname][end].Λ > _r
      tod = Pumas.Roots.find_zero(t -> cs(t) - _r, (obstimes[1], obstimes[end]))
      censored = 1
    else
      tod = obstimes[end]
      censored = 0
    end
    t_max_id = findlast(<=(tod), s.time)
    times = s.time[1:t_max_id]
    obs = map(s.observations) do x
      if eltype(x) <: TimeToEvent
        vcat(fill(missing, length(times)), censored)
      else
        vcat(x[1:t_max_id], missing)
      end
    end
    times = vcat(times, times[end] == tod ? tod + 1e-4 : tod)
    events = doses === nothing ? nothing : filter(x -> x.time <= tod, s.subject.events)
    Subject(
      s.subject.id,
      obs,
      s.subject.covariates,
      events,
      times
    )
  end
  return full_pop, sim
end

# rng = StableRNGs.StableRNG(111)
# ## Make different dosing regimens for each patient
# drs = [DosageRegimen(Float64(rand(rng, 1:3)), addl=rand(rng, 1:10), ii=0.05 + 0.1 * rand(rng)) for _ in 1:800]
# no_treatment = fill(DosageRegimen(1e-10), 200)
# dr = shuffle(StableRNGs.StableRNG(10), vcat(drs, no_treatment))

# _pop, datasim = generate_os_data(datamodel, dr; nsubj=length(dr), obstimes=(0:12.0) ./ 12, rng=StableRNGs.StableRNG(123))