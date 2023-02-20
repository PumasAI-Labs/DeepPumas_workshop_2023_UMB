using DeepPumas
using DeepPumas.SimpleChains
using StableRNGs
using CairoMakie
using Distributions
using Random

# 
# TABLE OF CONTENTS
# 
# 1. A SIMPLE MACHINE LEARNING (ML) MODEL
#
# 1.1. Sample subjects with an obvious `true_function`
# 1.2. Model `true_function` with a linear regression
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
# 4. INSPECTION OF THE VALIDATION LOSS AND REGULARIZATION
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
# 1. A SIMPLE MACHINE LEARNING (ML) MODEL
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

begin
    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    )
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 1.2. Model `true_function` with a linear regression

model_ex1 = @model begin
    @param begin
        a ∈ RealDomain()
        b ∈ RealDomain()
        σ ∈ RealDomain(; lower = 0.0)
    end
    @covariates x
    @pre ŷ = a * x + b
    @derived y ~ @. Normal(ŷ, σ)
end

fpm = fit(model_ex1, population_ex1, init_params(model_ex1), NaivePooled());
fpm  # `true_function` is y = x (that is, a = 1 b = 0) and σ = 0.25

ŷ = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

begin
    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    )
    scatter!(x, ŷ, label = "prediction")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

#
# 2. CAPTURING COMPLEX RELATIONSHIPS
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

begin
    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    )
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 2.2. Exercise: Reason about using a linear regression to model the current `true_function`

solution_ex22 = begin
    fpm = fit(model_ex1, population_ex2, init_params(model_ex1), MAP(NaivePooled()))
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
        nn ∈ MLP(1, (8, tanh), (1, identity); bias = true)
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
    NaivePooled();
    optim_options = (; iterations = 100),
);
fpm  # try to make sense of the parameters in the NN

ŷ_ex23 = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

begin
    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    )
    scatter!(x, ŷ_ex23, label = "prediction")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

#
# 3. BASIC UNDERFITTING AND OVERFITTING
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
        NaivePooled();
        optim_options = (; iterations = 10),
    )
    ŷ_underfit =
        [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

    fpm = fit(
        model_ex2,
        population_ex2,
        init_params(model_ex2),
        NaivePooled();
        optim_options = (; iterations = 5_000),
    )
    ŷ_overfit =
        [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    )
    scatter!(x, ŷ_underfit, label = "prediction (10 iterations)")
    scatter!(x, ŷ_ex23, label = "prediction (100 iterations)")
    scatter!(x, ŷ_overfit, label = "prediction (5k iterations)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?
#      Investigate the effect of `max_iterations`.

solution_ex32 = begin
    max_iterations = 10
    fpm = fit(
        model_ex1,
        population_ex2,
        init_params(model_ex1),
        NaivePooled();
        optim_options = (; iterations = max_iterations),
    )
    ŷ = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    )
    scatter!(x, ŷ, label = "prediction ($max_iterations iterations)")
    scatter!(x, ŷ_ex22, label = "prediction (exercise 2.2)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

# 3.3. The impact of the NN size

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
    NaivePooled();
    optim_options = (; iterations = 1000),
);

ŷ = [only(subject_prediction.pred.y) for subject_prediction in predict(fpm)]

begin
    f = scatter(
        x,
        y;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "data (each dot is a subject)",
    )
    scatter!(x, ŷ, label = "prediction (32x32 units - 1k iter)")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

#
# 4. INSPECTION OF THE VALIDATION LOSS AND REGULARIZATION
#
# 4.1. Validation loss as a proxy for generalization performance
# 4.2. Regularization to prevent overfitting
# 

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

begin
    f = scatter(
        x_train,
        y_train;
        axis = (xlabel = "covariate x", ylabel = "observation y"),
        label = "training data",
    )
    scatter!(x_valid, y_valid; label = "validation data")
    lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true")
    axislegend()
    f
end

begin
    loss_train_l, loss_valid_l = [], []

    fpm = fit(
        model_ex3,
        population_train,
        init_params(model_ex3),
        NaivePooled();
        optim_options = (; iterations = 10),
    )
    push!(loss_train_l, cost(model_ex3, population_train, coef(fpm), nothing, mse))
    push!(loss_valid_l, cost(model_ex3, population_valid, coef(fpm), nothing, mse))

    iteration_blocks = 100
    for _ = 2:iteration_blocks
        fpm = fit(
            model_ex3,
            population_train,
            coef(fpm),
            MAP(NaivePooled());
            optim_options = (; iterations = 10),
        )

        push!(loss_train_l, cost(model_ex3, population_train, coef(fpm), nothing, mse))
        push!(loss_valid_l, cost(model_ex3, population_valid, coef(fpm), nothing, mse))
    end
end

begin
    f, ax = scatterlines(
        1:iteration_blocks,
        Float32.(loss_train_l);
        label = "training",
        axis = (; xlabel = "Blocks of 10 iterations", ylabel = "Mean squared loss"),
    )
    scatterlines!(1:iteration_blocks, Float32.(loss_valid_l); label = "validation")
    axislegend()
    f
end

# 4.2. Regularization to prevent overfitting

model_ex4 = @model begin
    @param begin
        nn ∈ MLP(1, (32, tanh), (32, tanh), (1, identity); bias = true, reg = L2(1.0))
        σ ∈ RealDomain(; lower = 0.0)
    end
    @covariates x
    @pre ŷ = only(nn(x))
    @derived y ~ @. Normal(ŷ, σ)
end

begin
    reg_loss_train_l, reg_loss_valid_l = [], []

    fpm = fit(
        model_ex4,
        population_train,
        init_params(model_ex4),
        MAP(NaivePooled());
        optim_options = (; iterations = 10),
    )

    push!(reg_loss_train_l, cost(model_ex4, population_train, coef(fpm), nothing, mse))
    push!(reg_loss_valid_l, cost(model_ex4, population_valid, coef(fpm), nothing, mse))

    iteration_blocks = 100
    for _ = 2:iteration_blocks
        fpm = fit(
            model_ex4,
            population_train,
            coef(fpm),
            MAP(NaivePooled());
            optim_options = (; iterations = 10),
        )

        push!(reg_loss_train_l, cost(model_ex4, population_train, coef(fpm), nothing, mse))
        push!(reg_loss_valid_l, cost(model_ex4, population_valid, coef(fpm), nothing, mse))
    end
end

begin
    f, ax = scatterlines(
        1:iteration_blocks,
        Float32.(loss_train_l);
        label = "training",
        axis = (; xlabel = "Blocks of 10 iterations", ylabel = "Mean squared loss"),
    )
    scatterlines!(1:iteration_blocks, Float32.(loss_valid_l); label = "validation")
    scatterlines!(1:iteration_blocks, Float32.(reg_loss_train_l); label = "training (L2)")
    scatterlines!(1:iteration_blocks, Float32.(reg_loss_valid_l); label = "validation (L2)")
    axislegend()
    f
end
