"""
The matrix gradient of a time evolution block.

TODO: add Add backward, the hamiltonian backward.
"""
function tegrad(::Type{T}, te::TimeEvolution, y) where T
    error()
    -im*(mat(te.H)*y)
end

function mat_back(::Type{T}, te::TimeEvolution, y, adjy, collector) where T
    projection(te.dt, sum(adjy .* tegrad(T, te, y)))
end
