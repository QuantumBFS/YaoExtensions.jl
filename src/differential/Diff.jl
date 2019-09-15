export Rotor, generator, Diff, backward!, classical_autodiff!, CPhaseGate, DiffBlock
export dispatch_to_diff!, parameters_of_diff
import Yao: expect, content, chcontent, mat, apply!
using StatsBase

using Yao

############# General Rotor ############
#################### The Basic Diff #################
"""
    Diff{GT, N} <: TagBlock{GT, N}
    Diff(block) -> Diff

Mark a block as quantum differentiable.
"""
struct Diff{GT, N} <: TagBlock{GT, N}
    block::GT
    Diff(block::AbstractBlock{N}) where {N} = new{typeof(block), N}(block)
end
content(cb::Diff) = cb.block
chcontent(cb::Diff, blk::AbstractBlock) = Diff(blk)
YaoBlocks.PropertyTrait(::Diff) = YaoBlocks.PreserveAll()

apply!(reg::AbstractRegister, db::Diff) = apply!(reg, content(db))
mat(::Type{T}, df::Diff) where T = mat(T, df.block)
Base.adjoint(df::Diff) = chcontent(df, content(df)')

function YaoBlocks.print_annotation(io::IO, df::Diff)
    printstyled(io, "[∂] "; bold=true, color=:yellow)
end

#### interface #####
export autodiff, numdiff, opdiff, StatFunctional, statdiff, as_weights

as_weights(probs::AbstractVector{T}) where T = Weights(probs, T(1))
"""
    autodiff(mode::Symbol, block::AbstractBlock) -> AbstractBlock
    autodiff(mode::Symbol) -> Function

automatically mark differentiable items in a block tree as differentiable.
"""
function autodiff end
autodiff(mode::Symbol) = block->autodiff(mode, block)
autodiff(mode::Symbol, block::AbstractBlock) = autodiff(Val(mode), block)

# for BP
# Sequential, Roller and ChainBlock can propagate.
function autodiff(mode::Val{:BP}, blk::Union{ChainBlock, Sequential})
    chsubblocks(blk, autodiff.(mode, subblocks(blk)))
end

# for QC
autodiff(::Val{:QC}, block::Union{RotationGate, CphaseGate}) = Diff(block)
# escape control blocks.
autodiff(::Val{:QC}, block::ControlBlock) = block

function autodiff(mode::Val{:QC}, blk::AbstractBlock)
    blks = subblocks(blk)
    isempty(blks) ? blk : chsubblocks(blk, autodiff.(mode, blks))
end

@inline function _perturb(func, gate::Diff{<:DiffBlock}, δ::Real)
    dispatch!(-, gate, (δ,))
    r1 = func()
    dispatch!(+, gate, (2δ,))
    r2 = func()
    dispatch!(-, gate, (δ,))
    r1, r2
end

@inline function _perturb(func, gate::Diff{<:Rotor}, δ::Real)  # for put
    dispatch!(-, gate, (δ,))
    r1 = func()
    dispatch!(+, gate, (2δ,))
    r2 = func()
    dispatch!(-, gate, (δ,))
    r1, r2
end

"""
    numdiff(loss, diffblock::Diff; δ::Real=1e-2)

Numeric differentiation.
"""
@inline function numdiff(loss, diffblock::Diff; δ::Real=1e-2)
    r1, r2 = _perturb(loss, diffblock, δ)
    (r2-r1)/2δ
end

"""
    opdiff(psifunc, diffblock::Diff, op::AbstractBlock)

Operator differentiation.
"""
@inline function opdiff(psifunc, diffblock::Diff, op::AbstractBlock)
    r1, r2 = _perturb(()->expect(op, psifunc()) |> real, diffblock, π/2)
    (r2 - r1)/2
end

"""
    StatFunctional{N, AT}
    StatFunctional(array::AT<:Array) -> StatFunctional{N, <:Array}
    StatFunctional{N}(func::AT<:Function) -> StatFunctional{N, <:Function}

statistic functional, i.e.
    * if `AT` is an array, A[i,j,k...], it is defined on finite Hilbert space, which is `∫A[i,j,k...]p[i]p[j]p[k]...`
    * if `AT` is a function, F(xᵢ,xⱼ,xₖ...), this functional is `1/C(r,n)... ∑ᵢⱼₖ...F(xᵢ,xⱼ,xₖ...)`, see U-statistics for detail.

References:
    U-statistics, http://personal.psu.edu/drh20/asymp/fall2006/lectures/ANGELchpt10.pdf
"""
struct StatFunctional{N, AT}
    data::AT
    StatFunctional{N}(data::AT) where {N, AT<:Function} = new{N, AT}(data)
    StatFunctional(data::AT) where {N, AT<:AbstractArray{<:Real, N}} = new{N, AT}(data)
end

Base.parent(stat::StatFunctional) = stat.data

expect(stat::StatFunctional{2, <:AbstractArray}, px::Weights, py::Weights=px) = px.values' * stat.data * py.values
expect(stat::StatFunctional{1, <:AbstractArray}, px::Weights) = stat.data' * px.values
function expect(stat::StatFunctional{2, <:Function}, xs::AbstractVector{T}) where T
    N = length(xs)
    res = zero(stat.data(xs[1], xs[1]))
    for i = 2:N
        for j = 1:i-1
            @inbounds res += stat.data(xs[i], xs[j])
        end
    end
    res/binomial(N,2)
end
function expect(stat::StatFunctional{2, <:Function}, xs::AbstractVector, ys::AbstractVector)
    M = length(xs)
    N = length(ys)
    ci = CartesianIndices((M, N))
    @inbounds mapreduce(ind->stat.data(xs[ind[1]], ys[ind[2]]), +, ci)/M/N
end
expect(stat::StatFunctional{1, <:Function}, xs::AbstractVector) = mean(stat.data.(xs))
Base.ndims(stat::StatFunctional{N}) where N = N

"""
    statdiff(probfunc, diffblock::Diff, stat::StatFunctional{<:Any, <:AbstractArray}; initial::AbstractVector=probfunc())
    statdiff(samplefunc, diffblock::Diff, stat::StatFunctional{<:Any, <:Function}; initial::AbstractVector=samplefunc())

Differentiation for statistic functionals.
"""
@inline function statdiff(probfunc, diffblock::Diff, stat::StatFunctional{2}; initial::AbstractVector=probfunc())
    r1, r2 = _perturb(()->expect(stat, probfunc(), initial), diffblock, π/2)
    (r2 - r1)*ndims(stat)/2
end
@inline function statdiff(probfunc, diffblock::Diff, stat::StatFunctional{1})
    r1, r2 = _perturb(()->expect(stat, probfunc()), diffblock, π/2)
    (r2 - r1)*ndims(stat)/2
end

function classical_autodiff!(circuit::AbstractBlock, out::ArrayReg, adjout::ArrayReg; output_eltype=Any)
    collector = output_eltype[]
    in, inδ = backward!((out, adjout), circuit, collector)
    return 2*inδ, collector
end

dispatch_to_diff!(c::AbstractBlock, params) = dispatch_to_diff!(nothing, c, params)
function dispatch_to_diff!(f,c::AbstractBlock, params)
    dis = YaoBlocks.Dispatcher(params)
    postwalk(c) do blk
        blk isa Diff && dispatch!(f,blk, dis)
    end
end

function parameters_of_diff!(out, c::AbstractBlock)
    postwalk(c) do blk
        blk isa Diff && parameters!(out, blk)
    end
    return out
end
parameters_of_diff(c::AbstractBlock) = parameters_of_diff!(Float64[], c)

# to make a mat block differentiable
Yao.niparams(x::GeneralMatrixBlock{N,N}) where N = 1<<2N
Yao.getiparams(x::GeneralMatrixBlock) where N = (vec(x.mat)...,)

struct Echo{N,OT<:IO} <: PrimitiveBlock{N}
    sym::Symbol
    io::OT
end

Echo(nbits::Int, sym::Symbol) = Echo{nbits, typeof(stdout)}(sym, stdout)

Yao.apply!(reg::AbstractRegister, ec::Echo) = (println(ec.io, "apply!(::$(typeof(reg)), $(ec.sym))"); reg)
Yao.mat(reg::AbstractRegister, ec::Echo{N}) where N = (println(io, "mat($(ec.sym))"); IMatrix{1<<N})
