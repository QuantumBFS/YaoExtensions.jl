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
    throw(MethodError(mat_back!, (rb, adjy)))
end

function mat_back!(::Type{T}, rb::PutBlock{N, C, RT}, adjy, collector) where {T, N, C, RT}
    nparameters(rb) == 0 && return collector
    adjm = adjcunmat(adjy, N, (), (), mat(content(rb)), rb.locs)
    mat_back!(T, content(rb), adjm, collector)
end

function mat_back!(::Type{T}, rb::Concentrator{N}, adjy, collector) where {T,N}
    nparameters(rb) == 0 && return collector
    adjm = adjcunmat(adjy, N, (), (), mat(content(rb)), rb.locs)
    mat_back!(T, content(rb), adjm, collector)
end

function mat_back!(::Type{T}, rb::CachedBlock, adjy, collector) where T
    mat_back!(T, content(rb), adjy, collector)
end

function mat_back!(::Type{T}, rb::Daggered, adjy, collector) where T
    mat_back!(T, content(rb), adjy', collector)
end

function mat_back!(::Type{T}, rb::ControlBlock{N, C, RT}, adjy, collector) where {T, N, C, RT}
    nparameters(rb) == 0 && return collector
    adjm = adjcunmat(adjy, N, rb.ctrl_locs, rb.ctrl_config, mat(content(rb)), rb.locs)
    mat_back!(T, content(rb), adjm, collector)
end

function mat_back!(::Type{T}, rb::ChainBlock{N}, adjy, collector) where {T,N}
    np = nparameters(rb)
    np == 0 && return collector
    blocks = rb.blocks[end:-1:1]
    mi = mat(blocks[1])
    cache = Any[mi]
    for b in blocks[1:end-1]
        mi = mi*mat(b)
        push!(cache, mi)
    end
    for ib in length(rb):-1:1
        b = blocks[ib]
        adjb = ib==1 ? adjy : cache[ib-1]'*adjy
        mat_back!(T, b, adjb, collector)
        ib!=1 && (adjy = adjy*mat(b)')
    end
    collector[1:np] .= collector[np:-1:1]
    return collector
end

function mat_back!(::Type{T}, rb::KronBlock{N}, adjy, collector) where {T,N}
    nparameters(rb) == 0 && return collector
    for loc in rb.locs
        adjm = adjcunmat(adjy, N, (), (), mat(T,rb[loc]), (rb.locs...,))
        mat_back!(T, rb[loc], adjm, collector)
    end
    return collector
end

function mat_back!(::Type{T}, rb::Add{N}, adjy, collector) where {T,N}
    nparameters(rb) == 0 && return collector
    for b in subblocks(rb)[end:-1:1]
        mat_back!(T, b, adjy, collector)
    end
    return collector
end

function mat_back!(::Type{T}, rb::Scale{N}, adjy, collector) where {T,N}
    np = nparameters(rb)
    np == 0 && return collector
    mat_back!(T, content(rb), factor(rb)*adjy, collector)
    return collector
end
