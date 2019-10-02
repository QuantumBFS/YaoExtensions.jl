export inverselines, invorder_firstdim
"""
    inverselines(nbit::Int; n_reg::Int=nbit) -> ChainBlock

inverse first `n_reg` lines

TODO:
deprecate this function, it is not used.
"""
function inverselines(nbit::Int; n_reg::Int=nbit)
    c = chain(nbit)
    for i = 1:(n_reg ÷ 2)
        push!(c, swap(i,(n_reg-i+1)))
    end
    c
end

"""
    invorder_firstdim(v::VecOrMat) -> VecOrMat

inverse the bit order of first dimension.
"""
function invorder_firstdim(v::Matrix)
    w = similar(v)
    n = size(v, 1) |> log2i
    n_2 = n ÷ 2
    mask = [bmask(i, n-i+1) for i in 1:n_2]
    @simd for b in basis(n)
        @inbounds w[breflect(b, mask; nbits=n)+1,:] = v[b+1,:]
    end
    w
end

function invorder_firstdim(v::Vector)
    n = length(v) |> log2i
    n_2 = n ÷ 2
    w = similar(v)
    #mask = SVector{n_2, Int}([bmask(i, n-i+1)::Int for i in 1:n_2])
    mask = [bmask(i, n-i+1)::Int for i in 1:n_2]
    @simd for b in basis(n)
        @inbounds w[breflect(b, mask; nbits=n)+1] = v[b+1]
    end
    w
end
