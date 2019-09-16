Base.zero(pm::PermMatrix) = PermMatrix(pm.perm, zero(pm.vals))

function Random.randn!(m::Diagonal)
    randn!(m.diag)
    return m
end

function Random.randn!(m::SparseMatrixCSC)
    randn!(m.nzval)
    return m
end

function Random.randn!(m::PermMatrix)
    randn!(m.vals)
    return m
end

# TODO
# to make a mat block differentiable
#YaoBlocks.niparams(x::GeneralMatrixBlock{N,N}) where N = 1<<2N
#YaoBlocks.getiparams(x::GeneralMatrixBlock) where N = (vec(x.mat)...,)

# to make a scale block differentiable
#YaoBlocks.niparams(x::Scale{<:Number}) = 1
#YaoBlocks.getiparams(x::Scale{<:Number}) = factor(x)
