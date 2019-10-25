export Diff
"""
    Diff{GT, N} <: TagBlock{GT, N}
    Diff(block) -> Diff

Mark a block as quantum differentiable.
"""
struct Diff{GT, N} <: TagBlock{GT, N}
    content::GT
    function Diff(content::AbstractBlock{N}) where {N}
        @warn "Diff block has been deprecated, please use `Yao.AD.NoParams` to block non-differential parameters."
        new{typeof(content), N}(content)
    end
end
content(cb::Diff) = cb.content
chcontent(cb::Diff, blk::AbstractBlock) = Diff(blk)
YaoBlocks.PropertyTrait(::Diff) = YaoBlocks.PreserveAll()

apply!(reg::AbstractRegister, db::Diff) = apply!(reg, content(db))
mat(::Type{T}, df::Diff) where T = mat(T, df.content)
Base.adjoint(df::Diff) = chcontent(df, content(df)')

function YaoBlocks.print_annotation(io::IO, df::Diff)
    printstyled(io, "[∂] "; bold=true, color=:yellow)
end

#### interface #####
export markdiff

"""
    markdiff(mode::Symbol, block::AbstractBlock) -> AbstractBlock
    markdiff(mode::Symbol) -> Function

automatically mark differentiable items in a block tree as differentiable.
"""
function markdiff end

# for QC
markdiff(block::Union{RotationGate, CPhaseGate}) = Diff(block)
# escape control blocks.
markdiff(block::ControlBlock) = block

function markdiff(blk::AbstractBlock)
    blks = subblocks(blk)
    isempty(blks) ? blk : chsubblocks(blk, markdiff.(blks))
end

YaoBlocks.AD.mat_back!(::Type{T}, db::Diff, adjm::AbstractMatrix, collector) where T = AD.mat_back!(T, content(db), adjm, collector)
