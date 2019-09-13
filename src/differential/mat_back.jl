############### Primitive
"""
The matrix gradient of a rotation block.
"""
function rotgrad(::Type{T}, rb::RotationGate{N}) where {N, T}
    -sin(rb.theta / 2)/2 * IMatrix{1<<N}() + im/2 * cos(rb.theta / 2) * conj(mat(T, rb.block))
end

function mat_back!(::Type{T}, rb::RotationGate{N, RT}, adjy, collector) where {T, N, RT}
    pushfirst!(collector, projection(rb.theta, sum(adjy .* rotgrad(T, rb))))
end

function mat_back!(::Type{T}, A::GeneralMatrixBlock, adjy, collector) where T
    pushfirst!(collector, projection(A.mat, adjy))
end

function mat_back!(::Type{T}, rb::PhaseGate, adjy, collector) where {T}
    s = exp(-im*rb.theta)
    res = -1im*(adjy[1,1]*s + adjy[2,2]*s)
    pushfirst!(collector,projection(rb.theta, res))
end

function mat_back!(::Type{T}, rb::ShiftGate, adjy, collector) where {T}
    res = -1im*adjy[2,2] * exp(-1im*rb.theta)
    pushfirst!(collector, projection(rb.theta, res))
end

######################## Composite
function mat_back!(::Type{T}, rb::AbstractBlock, adjy, collector) where T
    nparameters(rb) == 0 && return collector
    throw(MethodError(mat_back, (rb, adjy)))
end

function mat_back!(::Type{T}, rb::PutBlock{N, C, RT}, adjy, collector) where {T, N, C, RT}
    nparameters(rb) == 0 && return collector
    adjm = adjcunmat(adjy, N, (), (), mat(content(rb)), rb.locs)
    mat_back!(T, content(rb), adjm, collector)
end

function mat_back!(::Type{T}, rb::ControlBlock{N, C, RT}, adjy, collector) where {T, N, C, RT}
    nparameters(rb) == 0 && return collector
    adjm = adjcunmat(adjy, N, rb.ctrl_locs, rb.ctrl_config, mat(content(rb)), rb.locs)
    mat_back!(T, content(rb), adjm, collector)
end

function mat_back!(::Type{T}, rb::ChainBlock{N}, adjy, collector) where {T,N}
    nparameters(rb) == 0 && return collector
    cache = Any[]
    m = mat(blk)
    push!(mat)
    for blk in Iterators.reverse(rb.blocks)
    end
    c = []
    for b in rb
        push!(c, b)
    end
    for b in rb
        mat_back!(T, b, adjy, collector)
    end
end

function mat_back!(::Type{T}, rb::KronBlock{N}, adjy, collector) where {T,N}
    for loc in rb.locs
        adjcunmat(adjy, N, (), (), mat(content(rb)), rb.locs)
    end
end
