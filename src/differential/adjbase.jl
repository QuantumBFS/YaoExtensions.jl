# data projection
#=
function *(sp::Union{SDSparseMatrixCSC, SDDiagonal, SDPermMatrix}, v::AbstractVector)
    sp*v, adjy -> (outer_projection(sp, adjy, v'), sp'*adjy)
end

function *(v::LinearAlgebra.Adjoint{T, V}, sp::Union{SDSparseMatrixCSC, SDDiagonal, SDPermMatrix}) where {T, V<:AbstractVector}
    v*sp, adjy -> (adjy*sp', outer_projection(sp, v', adjy))
end

function *(v::LinearAlgebra.Adjoint{T, V}, sp::SDDiagonal, v2::AbstractVector) where {T, V<:AbstractVector}
    v*sp, adjy -> (adjy*(sp*v2)', adjy*projection(sp, v', v2'), adjy*(v*sp)')
end
=#

function outer_projection(y::AbstractMatrix, adjy, v)
    out = zero(y)
    outer_projection!(out, adjy, v)
end

function outer_projection!(out::SparseMatrixCSC, adjy, v)
    # adjy*v^T
    is, js, vs = findnz(out)
    for (k,(i,j)) in enumerate(zip(is, js))
        @inbounds out.nzval[k] += adjy[i]*v[j]
    end
    out
end

outer_projection!(y::Diagonal, adjy, v) = (y.diag .+= adjy.*vec(v); y)
function outer_projection!(y::PermMatrix, adjy, v)
    for i=1:size(y, 1)
        @inbounds y.vals[i] += adjy[i] * v[y.perm[i]]
    end
    y
end
outer_projection!(y::Matrix, adjy, v) = y .+= adjy.*v

"""
Project a dense matrix to a sparse matrix
"""
@inline function projection(y::AbstractMatrix, m::AbstractMatrix)
    projection!(zero(y), m)
end

@inline function projection!(y::AbstractSparseMatrix, m::AbstractMatrix)
    @show y, m
    is, js, vs = findnz(y)
    for (k,(i,j)) in enumerate(zip(is, js))
        y.nzval[k] = m[i,j]
    end
    out
end

Base.zero(pm::PermMatrix) = PermMatrix(pm.perm, zero(pm.vals))
projection!(y::Diagonal, m::AbstractMatrix) = (y.diag .= diag(m); y)
@inline function projection!(y::PermMatrix, m::AbstractMatrix)
    for i=1:size(y, 1)
        @inbounds y.vals[i] = m[i,y.perm[i]]
    end
    res
end
projection(x::RT, adjx::Complex) where RT<:Real = RT(real(adjx))
projection(x::T, y::T) where T = y
projection(x::T1, y::T2) where {T1, T2} = convert(T1, y)

# fix kron in Zygote if having time
#function kron_back(a::AbstractMatrix, b::AbstractMatrix, adjy)
#end

function Random.randn!(m::Diagonal)
    target = zero(m)
    randn!(target.diag)
    return m
end

function Random.randn!(m::SparseMatrixCSC)
    target = zero(m)
    randn!(target.nzval)
    return m
end

function Random.randn!(m::PermMatrix)
    target = zero(m)
    randn!(target.vals)
    return m
end

struct OuterProduct{T} <: AbstractMatrix{T}
    left::Vector{T}
    right::Vector{T}
end

Base.getindex(op::OuterProduct, i::Int, j::Int) = op.left[i]*op.right[j]
Base.size(op::OuterProduct) = (length(op.left), length(op.right))
Base.size(op::OuterProduct, i::Int) = i==1 ? length(op.left) : (i==2 ? length(op.right) : throw(DimensionMismatch("")))
