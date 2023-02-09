using Revise
using SimpleChains
using Distributions
using Random
using CairoMakie
using Printf

# regression example

"""
    sample_regression(num_samples; true_function=x->x, stdev=0.25, seed=nothing)

Sample `num_samples` pairs (xᵢ, yᵢ), where ``xᵢ ~ U(-1, 1)``, ``ϵᵢ ~ N(0, stdev)``, and 
``yᵢ = true_function(xᵢ) + ϵᵢ``.

A random `seed` can be optionally passed. Otherwise, none is set.

# Examples
```jldoctest
julia> sample_regression(2; seed=1)
([-0.8532672910614143, -0.3015170208856277], [-1.0549803725630928, 0.31273081261019553])
```
"""
function sample_regression(
    num_samples::Integer;
    true_function::Function=x->x,
    stdev::Real=0.25, 
    seed=nothing
)

    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    x = rand(Uniform(-1, 1), num_samples)
    ϵ = rand(Normal(0, stdev), num_samples)

    return x, true_function.(x) .+ ϵ
end

x, y = sample_regression(50; seed=1)
f = scatter(x, y; color=:gray, label="data");
ablines!(0, 1; label="intercept = 0, slope = 1 (true)")
f

xs = reshape(x, 1, :)
ys = reshape(y, 1, :)

linreg = SimpleChain(
  static(1),
  TurboDense{true}(identity, 1)
)

p = SimpleChains.init_params(linreg)
G = SimpleChains.alloc_threaded_grad(linreg)

loss = SimpleChains.add_loss(linreg, SquaredLoss(ys))
SimpleChains.train_unbatched!(G, p, loss, xs, SimpleChains.ADAM(), 10_000)

fitted_slope, fitted_intercept = p[1], p[2]
ablines!(
  fitted_intercept, fitted_slope; 
  label="intercept = $(@sprintf("%.2f", fitted_intercept)), slope = $(@sprintf("%.2f", fitted_slope)) (fitted)"
)
axislegend()
f


# can not have a simple classification example because SimpleChains doesn't have BCE yet

y_binarized = UInt32.(Int.(y .>= mean(y)) .+ 1)

f = scatter(x, y_binarized; color=:gray, label="data");
f

# workaround with a 2-class CCE, but that's not intuitive...
logregish = SimpleChain(
  static(1),
  TurboDense{true}(identity, 2)
)


p = SimpleChains.init_params(logregish)
G = SimpleChains.alloc_threaded_grad(logregish)

loss = SimpleChains.add_loss(logregish, LogitCrossEntropyLoss(y_binarized))
SimpleChains.train_unbatched!(G, p, loss, xs, SimpleChains.ADAM(), 10_000)

preds = map(argmax, eachcol(logregish(xs, p))) 
scatter!(
  x, preds; label="predictions"
)
axislegend()
f