using LinearAlgebra: Adjoint
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

function YaoBlocks.expect(stat::StatFunctional{2}, xs::AbstractVector{T}) where T
    N = length(xs)
    res = zero(stat.f(xs[1], xs[1]))
    for i = 2:N
        for j = 1:i-1
            @inbounds res += stat.f(xs[i], xs[j])
        end
    end
    res/binomial(N,2)
end

function YaoBlocks.expect(stat::StatFunctional{2}, xs::AbstractVector, ys::AbstractVector)
    M = length(xs)
    N = length(ys)
    ci = CartesianIndices((M, N))
    @inbounds mapreduce(ind->stat.f(xs[ind[1]], ys[ind[2]]), +, ci)/M/N
end

YaoBlocks.expect(stat::StatFunctional{1}, xs::AbstractVector) = mean(stat.f.(xs))
Base.ndims(stat::StatFunctional{N}) where N = N

"""
    faithful_statdiff(stat::StatFunctional{2}, pair::Pair{<:ArrayReg,<:AbstractBlock})

Differentiation for statistic functionals.
"""
@inline function faithful_statdiff(stat::StatFunctional{2}, pair::Pair{<:ArrayReg,<:AbstractBlock})
    initial = copy(pair.first) |> pair.second
    map(get_diffblocks(pair.second)) do diffblock
        r1, r2 = _perturb(()->expect(stat, copy(pair.first) |> pair.second, initial), diffblock, π/2)
        (r2 - r1)*ndims(stat)/2
    end
end

@inline function faithful_statdiff(stat::StatFunctional{1}, pair::Pair{<:ArrayReg,<:AbstractBlock})
    map(get_diffblocks(pair.second)) do diffblock
        r1, r2 = _perturb(()->expect(stat, copy(pair.first) |> pair.second), diffblock, π/2)
        (r2 - r1)*ndims(stat)/2
    end
end

function (::Adjoint{Any,typeof(expect)})(stat::StatFunctional, circuit::Pair{<:ArrayReg, <:AbstractBlock})
    reg, c = circuit
    out = copy(reg) |> c
    outδ = copy(out) |> op
    (in, inδ), paramsδ = apply_back((out, outδ), c)
    return outδ => paramsδ.*2
end
