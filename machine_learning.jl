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
    y = true_function.(x) .+ ϵ

    idx = sortperm(x)
    x = x[idx]
    y = y[idx]

    return x, y
end

x, y = sample_regression(50; seed=1)
f = scatter(x, y; color=:gray, label="data");
lines!(x, x; label="true")
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
y_hat = linreg(xs, p)

lines!(x, y_hat[:]; label="predicted")
axislegend()
f

# EXAMPLE 2: MAKE true_function more COMPLEX

x, y = sample_regression(50; true_function=x -> x^2, seed=1)
f = scatter(x, y; color=:gray, label="data");
lines!(x, x.^2)
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
y_hat = linreg(xs, p)

lines!(x, y_hat[:]; label="predicted")
axislegend()
f

mlp = SimpleChain(
  static(1),
  TurboDense{true}(tanh, 8),
  TurboDense{true}(identity, 1)
)

p = SimpleChains.init_params(mlp)
G = SimpleChains.alloc_threaded_grad(mlp)

loss = SimpleChains.add_loss(mlp, SquaredLoss(ys))
SimpleChains.train_unbatched!(G, p, loss, xs, SimpleChains.ADAM(), 10_000)

y_hat = mlp(xs, p)

f = scatter(x, y; color=:gray, label="data");
lines!(x, true_function.(x))
lines!(x, y_hat[:]; label="predicted")
axislegend()
f

# EXAMPLE 3: OVERFITTING

# REPEAT SAME BUT Train much longer with same architecture for overfitting

mlp = SimpleChain(
  static(1),
  TurboDense{true}(tanh, 8),
  TurboDense{true}(identity, 1)
)

p = SimpleChains.init_params(mlp)
G = SimpleChains.alloc_threaded_grad(mlp)

loss = SimpleChains.add_loss(mlp, SquaredLoss(ys))
SimpleChains.train_unbatched!(G, p, loss, xs, SimpleChains.ADAM(), 10_000_000)

y_hat = mlp(xs, p)

f = scatter(x, y; color=:gray, label="data");
lines!(x, true_function.(x))
lines!(x, y_hat[:]; label="predicted")
axislegend()
f

# EXAMPLE 4: OVERFITTING BY MODEL CAPACITY AND TOO LONG TRAINING

mlp = SimpleChain(
  static(1),
  TurboDense{true}(tanh, 128),
  TurboDense{true}(tanh, 128),
  TurboDense{true}(identity, 1)
)

p = SimpleChains.init_params(mlp)
G = SimpleChains.alloc_threaded_grad(mlp)

loss = SimpleChains.add_loss(mlp, SquaredLoss(ys))
SimpleChains.train_unbatched!(G, p, loss, xs, SimpleChains.ADAM(), 1_000_000)

y_hat = mlp(xs, p)

f = scatter(x, y; color=:gray, label="data");
lines!(x, true_function.(x))
lines!(x, y_hat[:]; label="predicted")
axislegend()
f


# EXERCISE 5: CONTROLING VALIDATION LOSS

x_train, y_train = sample_regression(50; true_function=x -> x^2)
x_valid, y_valid = sample_regression(50; true_function=x -> x^2)

f = scatter(x_train, y_train; color=:gray, label="training");
scatter!(x_valid, y_valid; label="validation");
axislegend()
f

mlp = SimpleChain(
  static(1),
  TurboDense{true}(tanh, 128),
  TurboDense{true}(tanh, 128),
  TurboDense{true}(identity, 1)
)

xs_train, ys_train = reshape(x_train, 1, :), reshape(y_train, 1, :)
xs_valid, ys_valid = reshape(x_valid, 1, :), reshape(y_valid, 1, :)

p = SimpleChains.init_params(mlp)
G = SimpleChains.alloc_threaded_grad(mlp)

loss_train = SimpleChains.add_loss(mlp, SquaredLoss(ys_train))
loss_valid = SimpleChains.add_loss(mlp, SquaredLoss(ys_valid))

loss_train_l, loss_valid_l = [], []
epoch_range = 1:200
for epoch in epoch_range
  SimpleChains.train_unbatched!(G, p, loss_train, xs_train, SimpleChains.ADAM(), 100);
  push!(loss_train_l, loss_train(xs_train, p))
  push!(loss_valid_l, loss_valid(xs_valid, p))
end

f, ax = scatterlines(
  epoch_range, Float32.(loss_train_l); 
  label="training",
  axis = (; xlabel = "~Epochs", ylabel="Loss")
)
scatterlines!(epoch_range, Float32.(loss_valid_l); label="validation")
axislegend()
f

f, ax = scatterlines(
  epoch_range[1:20], Float32.(loss_train_l)[1:20]; 
  label="training",
  axis = (; xlabel = "~Epochs", ylabel="Loss")
)
scatterlines!(epoch_range[1:20], Float32.(loss_valid_l)[1:20]; label="validation")
axislegend()
f


# EXERCISE 6: CLASSIFICATION EXAMPLE

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