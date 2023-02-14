using DeepPumas
using DeepPumas: SimpleChainDomain, SimpleChains
using StableRNGs
using CairoMakie
using Distributions
using Random
using Printf
using JuliaFormatter

# 
# TABLE OF CONTENTS
# 
# 1. A SIMPLE MACHINE LEARNING (ML) MODEL
# 1.1. Sample subjects with an obvious `true_function`
# 1.2. Model `true_function` with a DeepPumas linear regression
#
# 2. CAPTURING COMPLEX RELATIONSHIPS
#
# 2.1. Sample subjects with a more complex `true_function`
# 2.2. Exercise: Reason about using a linear regression to model the current `true_function`
# 2.3. Use a neural network (NN) to model `true_function`
#
# 3. BASIC UNDERFITTING AND OVERFITTING
#
# 3.1. Exercise: Investigate the impact of the number of fitting iterations in NNs
#      (Hint: Train `model_ex2` on `population_ex2` for few and for many iterations.)
# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?
# 3.3. The impact of the NN size
# 
# EXAMPLE 4: INSPECTION OF THE VALIDATION LOSS AND REGULARIZATION
#
# 4.1. Validation loss as a proxy for generalization performance
# 4.2. Regularization to prevent overfitting
# 

"""
Helper Pumas model to generate synthetic data. Subjects will have one 
covariate `x`  and one observation `y ~ Normal(true_function(x), σ)`.
`true_function` and `σ` have to be defined independently, and the probability 
distribution of `x` has to be determined in the call to `synthetic_data`.
"""
data_model = @model begin
    @covariates x
    @pre x_ = x
    @derived begin
        y ~ @. Normal(true_function(x_), σ)
    end
end

#
# EXAMPLE 1: A SIMPLE MACHINE LEARNING (ML) MODEL
#
# 1.1. Sample subjects with an obvious `true_function`
# 1.2. Model `true_function` with a DeepPumas linear regression
#

# 1.1. Sample subjects with an obvious `true_function`

true_function = x -> x
σ = 0.25

population_ex1 = synthetic_data(
    data_model;
    covariates = (; x = Uniform(-1, 1)),
    obstimes = [0.0],
    rng = StableRNG(0),  # must use `StableRNGs` until bug fix in next release
)

x = [only(subject.covariates().x) for subject in population_ex1]
y = [only(subject.observations.y) for subject in population_ex1]

f = scatter(
    x,
    y;
    axis = (xlabel = "covariate x", ylabel = "observation y"),
    label = "data (each dot is a subject)",
);
lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true");
axislegend();
f

# 1.2. Model `true_function` with a DeepPumas linear regression

model_ex1 = @model begin
    @param begin
        linreg ∈ SimpleChainDomain(  # define linear regression y = a * x + b
            SimpleChain(
                static(1),  # one input
                TurboDense{true}(identity, 1),  # one output with intercept
            ),
        )
        σ ∈ RealDomain(; lower = 0.0)
    end
    @covariates x
    @pre ŷ = only(linreg(x))
    @derived y ~ @. Normal(ŷ, σ)
end

fpm = fit(
    model_ex1,
    population_ex1,
    init_params(model_ex1),
    MAP(NaivePooled());
    optim_options = (; iterations = 100),
);
fpm  # `true_function` is y = x (that is, a = 1 b = 0) and σ = 0.25

ŷ = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

f = scatter(
    x,
    y;
    axis = (xlabel = "covariate x", ylabel = "observation y"),
    label = "data (each dot is a subject)",
);
scatter!(x, ŷ, label = "prediction")
lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
axislegend()
f

#
# EXAMPLE 2: CAPTURING COMPLEX RELATIONSHIPS
#
# 2.1. Sample subjects with a more complex `true_function`
# 2.2. Exercise: Reason about using a linear regression to model the current `true_function`
# 2.3. Use a neural network (NN) to model `true_function`
#

# 2.1. Sample subjects with a more complex `true_function`

true_function = x -> x^2  # the examples aim to be insightful; please, play along!
σ = 0.25

population_ex2 = synthetic_data(
    data_model;
    covariates = (; x = Uniform(-1, 1)),
    obstimes = [0.0],
    rng = StableRNG(0),  # must use `StableRNGs` until bug fix in next release
)

x = [only(subject.covariates().x) for subject in population_ex2]
y = [only(subject.observations.y) for subject in population_ex2]

f = scatter(
    x,
    y;
    axis = (xlabel = "covariate x", ylabel = "observation y"),
    label = "data (each dot is a subject)",
);
lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true");
axislegend();
f

# 2.2. Exercise: Reason about using a linear regression to model the current `true_function`

solution_ex22 = begin
    fpm = fit(
        model_ex1,
        population_ex2,
        init_params(model_ex1),
        MAP(NaivePooled());
        optim_options = (; iterations = 100),
    )
    ŷ_ex22 = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    )
    scatter!(x, ŷ_ex22, label = "prediction (fpm)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 2.3. Use a neural network (NN) to model `true_function`

model_ex2 = @model begin
    @param begin  
        nn ∈ MLP(1, (8, tanh), (1, identity); bias = true)  # API supporting SimpleChains and Flux
        σ ∈ RealDomain(; lower = 0.0)
    end
    @covariates x
    @pre ŷ = only(nn(x))
    @derived y ~ @. Normal(ŷ, σ)
end

fpm = fit(
    model_ex2,
    population_ex2,
    init_params(model_ex2),
    MAP(NaivePooled());
    optim_options = (; iterations = 100),
);
fpm  # try to make sense of the parameters in the NN

ŷ_ex23 = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

f = scatter(
    x,
    y;
    axis = (xlabel = "covariate x", ylabel = "observation y"),
    label = "data (each dot is a subject)",
);
scatter!(x, ŷ_ex23, label = "prediction (fpm_ex23)")
lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
axislegend()
f

#
# EXAMPLE 3: BASIC UNDERFITTING AND OVERFITTING
#
# 3.1. Exercise: Investigate the impact of the number of fitting iterations in NNs
#      (Hint: Train `model_ex2` on `population_ex2` for few and for many iterations.)
# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?
# 3.3. The impact of the NN size
# 

# 3.1. Exercise: Investigate the impact of the number of fitting iterations in NNs
#      (Hint: Train `model_ex2` on `population_ex2` for few and for many iteration

solution_ex31 = begin
    fpm = fit(
        model_ex2,
        population_ex2,
        init_params(model_ex2),
        MAP(NaivePooled());
        optim_options = (; iterations = 10),
    );
    ŷ_underfit = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

    fpm = fit(
        model_ex2,
        population_ex2,
        init_params(model_ex2),
        MAP(NaivePooled());
        optim_options = (; iterations = 10_000),  #TODO: NOT RUNNING THIS LONG ACTUALLY
    );
    ŷ_overfit = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    );
    scatter!(x, ŷ_underfit, label = "prediction (10 iterations)")
    scatter!(x, ŷ_ex23, label = "prediction (100 iterations)")
    scatter!(x, ŷ_overfit, label = "prediction (10k iterations)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?

solution_ex32 = begin
    fpm = fit(
        model_ex1,
        population_ex2,
        init_params(model_ex1),
        MAP(NaivePooled());
        optim_options = (; iterations = 10),
    );
    ŷ_underfit = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

    fpm = fit(
        model_ex1,
        population_ex2,
        init_params(model_ex1),
        MAP(NaivePooled());
        optim_options = (; iterations = 10_000),  #TODO: NOT RUNNING THIS LONG ACTUALLY
    );
    ŷ_overfit = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    );
    scatter!(x, ŷ_underfit, label = "prediction (10 iterations)")
    scatter!(x, ŷ_ex22, label = "prediction (100 iterations)")
    scatter!(x, ŷ_overfit, label = "prediction (10k iterations)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 3.3. The impact of the NN size

#TODO: This in SimpleChains would be much faster with larger net
model_ex3 = @model begin
    @param begin  
        nn ∈ MLP(1, (32, tanh), (32, tanh), (1, identity); bias = true)
        σ ∈ RealDomain(; lower = 0.0)
    end
    @covariates x
    @pre ŷ = only(nn(x))
    @derived y ~ @. Normal(ŷ, σ)
end

fpm = fit(
    model_ex3,
    population_ex2,
    init_params(model_ex3),
    MAP(NaivePooled());
    optim_options = (; iterations = 1000),
);

ŷ = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

f = scatter(
    x,
    y;
    axis = (xlabel = "covariate x", ylabel = "observation y"),
    label = "data (each dot is a subject)",
);
scatter!(x, ŷ, label = "prediction (32x32 units - 1k iter)")
lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
axislegend()
f

#
# EXAMPLE 4: INSPECTION OF THE VALIDATION LOSS AND REGULARIZATION
#
# 4.1. Validation loss as a proxy for generalization performance
# 4.2. Regularization to prevent overfitting
# 
#TODO: Again, it is difficult to overfit. I find it strange.

# 4.1. Validation loss as a proxy for generalization performance

population_train = population_ex2
x_train, y_train = x, y

population_valid = synthetic_data(
    data_model;
    covariates = (; x = Uniform(-1, 1)),
    obstimes = [0.0],
    rng = StableRNG(1),  # must use `StableRNGs` until bug fix in next release
)
x_valid = [only(subject.covariates().x) for subject in population_valid]
y_valid = [only(subject.observations.y) for subject in population_valid]

f = scatter(
    x_train,
    y_train;
    axis = (xlabel = "covariate x", ylabel = "observation y"),
    label = "training data",
);
scatter!(x_valid, y_valid; label = "validation data");
lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true");
axislegend();
f

loss_train_l, loss_valid_l = [], []

fpm = fit(
    model_ex3,
    population_ex2,
    init_params(model_ex3),
    MAP(NaivePooled());
    optim_options = (; iterations = 10),
);

loss_train = SimpleChains.add_loss(coef(fpm).nn.model, SquaredLoss(y_train))
loss_valid = SimpleChains.add_loss(coef(fpm).nn.model, SquaredLoss(y_valid))

push!(loss_train_l, loss_train(reshape(x_train, 1, :), coef(fpm).nn.param))
push!(loss_valid_l, loss_valid(reshape(x_valid, 1, :), coef(fpm).nn.param))

iteration_blocks = 100
for _ in 2:iteration_blocks

    fpm = fit(
        model_ex3,
        population_ex2,
        coef(fpm),
        MAP(NaivePooled());
        optim_options = (; iterations = 10),
    );  

    push!(loss_train_l, loss_train(reshape(x_train, 1, :), coef(fpm).nn.param))
    push!(loss_valid_l, loss_valid(reshape(x_valid, 1, :), coef(fpm).nn.param))

end

f, ax = scatterlines(
    1:iteration_blocks,
    Float32.(loss_train_l);
    label = "training",
    axis = (; xlabel = "Blocks of 10 iterations", ylabel = "Squared loss"),
)
scatterlines!(1:iteration_blocks, Float32.(loss_valid_l); label = "validation")
axislegend()
f

# 4.2. Regularization to prevent overfitting

model_ex4 = @model begin
    @param begin  
        nn ∈ MLP(1, (32, tanh), (32, tanh), (1, identity); bias = true, reg = L2(1.))
        σ ∈ RealDomain(; lower = 0.0)
    end
    @covariates x
    @pre ŷ = only(nn(x))
    @derived y ~ @. Normal(ŷ, σ)
end

reg_loss_train_l, reg_loss_valid_l = [], []

fpm = fit(
    model_ex4,
    population_ex2,
    init_params(model_ex4),
    MAP(NaivePooled());
    optim_options = (; iterations = 10),
);

loss_train = SimpleChains.add_loss(coef(fpm).nn.model, SquaredLoss(y_train))
loss_valid = SimpleChains.add_loss(coef(fpm).nn.model, SquaredLoss(y_valid))

push!(reg_loss_train_l, loss_train(reshape(x_train, 1, :), coef(fpm).nn.param))
push!(reg_loss_valid_l, loss_valid(reshape(x_valid, 1, :), coef(fpm).nn.param))

iteration_blocks = 100
for _ in 2:iteration_blocks

    fpm = fit(
        model_ex4,
        population_ex2,
        coef(fpm),
        MAP(NaivePooled());
        optim_options = (; iterations = 10),
    );  

    push!(reg_loss_train_l, loss_train(reshape(x_train, 1, :), coef(fpm).nn.param))
    push!(reg_loss_valid_l, loss_valid(reshape(x_valid, 1, :), coef(fpm).nn.param))

end

f, ax = scatterlines(
    1:iteration_blocks,
    Float32.(loss_train_l);
    label = "training",
    axis = (; xlabel = "Blocks of 10 iterations", ylabel = "Squared loss"),
)
scatterlines!(1:iteration_blocks, Float32.(loss_valid_l); label = "validation")
scatterlines!(1:iteration_blocks, Float32.(reg_loss_train_l); label = "training (L2)")
scatterlines!(1:iteration_blocks, Float32.(reg_loss_valid_l); label = "validation (L2)")
axislegend()
f


#
# EXERCISE 6: CLASSIFICATION EXAMPLE
#

# can not have a simple classification example because SimpleChains doesn't have BCE yet

y_binarized = UInt32.(Int.(y .>= mean(y)) .+ 1)

f = scatter(x, y_binarized; color = :gray, label = "data");
f

# workaround with a 2-class CCE, but that's not intuitive...
logregish = SimpleChain(static(1), TurboDense{true}(identity, 2))


p = SimpleChains.init_params(logregish)
G = SimpleChains.alloc_threaded_grad(logregish)

loss = SimpleChains.add_loss(logregish, LogitCrossEntropyLoss(y_binarized))
SimpleChains.train_unbatched!(G, p, loss, xs, SimpleChains.ADAM(), 10_000)

preds = map(argmax, eachcol(logregish(xs, p)))
scatter!(x, preds; label = "predictions")
axislegend()
f