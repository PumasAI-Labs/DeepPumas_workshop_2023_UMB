"""
    saturating_function(x)

Starting from x=65, increase slowly from 1 up until 2.

# Examples
```jldoctest
julia> saturating_function.([55, 65, 75, 85, 95])
5-element Vector{Real}:
 1
 1
 1.6321205588285577
 1.8646647167633872
 1.9502129316321362
```
"""
function saturating_function(x::Real)
    x <= 65 ? 1 : 2 - exp((65 - x) / 10)
end

"""
    pair_plots(A, B)

Produce a plot with N x M subplots, where N is the number of rows in `A` 
and M is the number of rows in `B`, and such that subplot [ij] is the 
scatterplot of the i-th row of `A` and the `j`-th row of B.
"""
function pair_plots(
    A::AbstractMatrix,
    B::AbstractMatrix;
    xlabels = nothing,
    ylabels = nothing,
)
    if isnothing(xlabels)
        xlabels = ["x$i" for i in 1:size(A)[1]]
    end
    if isnothing(ylabels)
        ylabels = ["y$j" for j in 1:size(B)[1]]
    end

    fig = Figure()
    for i = 1:size(A)[1]
        for j = 1:size(B)[1]
            scatter(
                fig[j, i],  # makes the array of subplots more readable
                A[i, :],
                B[j, :],
                axis = (xlabel = xlabels[i], ylabel = ylabels[j]),
            )
        end
    end
    fig
end
