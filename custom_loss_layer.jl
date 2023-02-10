using SimpleChains

struct BinaryLogitCrossEntropyLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}
    targets::Y
end

struct BinaryLogitCrossEntropyLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}
    targets::Y
end

SimpleChains.target(loss::BinaryLogitCrossEntropyLoss) = loss.targets
(loss::BinaryLogitCrossEntropyLoss)(x::AbstractArray) = BinaryLogitCrossEntropyLoss(x)

function calculate_loss(loss::BinaryLogitCrossEntropyLoss, logits)
    y = loss.targets
    total_loss = zero(eltype(logits))
    for i in eachindex(y)
        p_i = inv(1 + exp(-logits[i]))
        y_i = y[i]
        total_loss -= y_i * log(p_i) + (1 - y_i) * (1 - log(p_i))
    end
    total_loss
end
function (loss::BinaryLogitCrossEntropyLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
end

function SimpleChains.layer_output_size(::Val{T}, sl::BinaryLogitCrossEntropyLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end
function SimpleChains.forward_layer_output_size(::Val{T}, sl::BinaryLogitCrossEntropyLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{BinaryLogitCrossEntropyLoss},
    _::Ptr,
    pu::Ptr{UInt8},
) where {T}
    loss = getfield(layers, 1)
    total_loss = calculate_loss(loss, previous_layer_output)
    y = loss.targets

    # Store the backpropagated gradient in the previous_layer_output array.
    for i in eachindex(y)
        sign_arg = 2 * y[i] - 1
        # Get the value of the last logit
        logit_i = previous_layer_output[i]
        previous_layer_output[i] = -(sign_arg * inv(1 + exp(sign_arg * logit_i)))
    end

    return total_loss, previous_layer_output, pu
end

using SimpleChains

model = SimpleChain(
    static(2),
    TurboDense(tanh, 32),
    TurboDense(tanh, 16),
    TurboDense(identity, 1)
)

batch_size = 64
X = rand(Float32, 2, batch_size)
Y = rand(Bool, batch_size)

parameters = SimpleChains.init_params(model);
gradients = SimpleChains.alloc_threaded_grad(model);

# Add the loss like any other loss type
model_loss = SimpleChains.add_loss(model, BinaryLogitCrossEntropyLoss(Y));

SimpleChains.valgrad!(gradients, model_loss, X, parameters)

epochs = 100
SimpleChains.train_unbatched!(gradients, parameters, model_loss, X, SimpleChains.ADAM(), epochs);