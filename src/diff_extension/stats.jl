using LinearAlgebra: Adjoint
using StatsBase
export as_weights
export StatFunctional, faithful_statdiff

"""
    StatFunctional{N, F}
    StatFunctional{N}(f::F) -> StatFunctional{N}

`f` is a function, `f(xᵢ,xⱼ,xₖ...)`, this functional is `1/C(r,n)... ∑ᵢⱼₖ...f(xᵢ,xⱼ,xₖ...)`, see U-statistics for detail.

References:
    U-statistics, http://personal.psu.edu/drh20/asymp/fall2006/lectures/ANGELchpt10.pdf
"""
struct StatFunctional{N, F}
    f::F
    StatFunctional{N}(f::F) where {N, F} = new{N, F}(f)
end
Base.ndims(stat::StatFunctional{N}) where N = N

"U-statistics of order 2."
function YaoBlocks.expect(stat::StatFunctional{2}, xs::AbstractVector{T}) where T<:BitStr
    N = length(xs)
    res = zero(stat.f(xs[1], xs[1]))
    for i = 2:N
        for j = 1:i-1
            @inbounds res += stat.f(xs[i], xs[j])
        end
    end
    res/binomial(N,2)
end

function YaoBlocks.expect(stat::StatFunctional{2}, xs::AbstractVector{T}, ys::AbstractVector{T}) where T<:BitStr
    ci = CartesianIndices((length(xs), length(ys)))
    @inbounds mapreduce(ind->stat.f(xs[ind[1]], ys[ind[2]]), +, ci)/length(ci)
end

function YaoBlocks.expect(stat::StatFunctional{2}, xs::AbstractVector{T}, px::Weights, ys::AbstractVector{T}, py::Weights) where T<:BitStr
    ci = CartesianIndices((length(xs), length(ys)))
    @inbounds mapreduce(ind->px[ind[1]]*stat.f(xs[ind[1]], ys[ind[2]])*py[ind[2]], +, ci)
end

function YaoBlocks.expect(stat::StatFunctional{2}, px::Weights, py::Weights=px)
    expect(stat, basis(BitStr64{log2dim1(px)}), px, basis(BitStr64{log2dim1(py)}), py)
end

YaoBlocks.expect(stat::StatFunctional{1}, xs::AbstractVector{<:BitStr}) = mean(stat.f.(xs))
function YaoBlocks.expect(stat::StatFunctional{1}, px::Weights)
    T = BitStr64{log2dim1(px)}
    mapreduce(i->stat.f(T(i-1)) * px[i], +, 1:length(px))
end

as_weights(probs::AbstractVector{T}) where T = Weights(probs, T(1))

"""
    faithful_statdiff(stat::StatFunctional{2}, pair::Pair{<:ArrayReg,<:AbstractBlock})

Differentiation for statistic functionals.
"""
@inline function faithful_statdiff(stat::StatFunctional{2}, pair::Pair{<:ArrayReg,<:AbstractBlock})
    initial = copy(pair.first) |> pair.second |> probs |> as_weights
    map(get_diffblocks(pair.second)) do diffblock
        r1, r2 = _perturb(()->expect(stat, copy(pair.first) |> pair.second |> probs |> as_weights, initial), diffblock, π/2)
        (r2 - r1)*ndims(stat)/2
    end
end

@inline function faithful_statdiff(stat::StatFunctional{1}, pair::Pair{<:ArrayReg,<:AbstractBlock})
    map(get_diffblocks(pair.second)) do diffblock
        r1, r2 = _perturb(()->expect(stat, copy(pair.first) |> pair.second |> probs |> as_weights), diffblock, π/2)
        (r2 - r1)*ndims(stat)/2
    end
end

function (::Adjoint{Any,typeof(expect)})(stat::StatFunctional, circuit::Pair{<:ArrayReg, <:AbstractBlock})
    reg, c = circuit
    out = copy(reg) |> c
    outδ = ArrayReg(4*witness(stat, out |> probs).*statevec(out))
    (in, inδ), paramsδ = apply_back((out, outδ), c)
    return outδ => paramsδ.*2
end

function witness(stat::StatFunctional{2}, probs)
    T = BitStr64{log2dim1(probs)}
    map(i->mapreduce(j->stat.f(i, T(j-1))*probs[j], +, 1:length(probs)), basis(T))
end

function witness(stat::StatFunctional{1}, probs)
    stat.f.(basis(BitStr64{log2dim1(probs)}))
end
