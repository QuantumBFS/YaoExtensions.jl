export heisenberg, transverse_ising

"""
    heisenberg(nbit::Int; periodic::Bool=true)

1D heisenberg hamiltonian, for its ground state, refer `PRB 48, 6141`.
"""
function heisenberg(nbit::Int; periodic::Bool=true)
    sx = i->put(nbit, i=>X)
    sy = i->put(nbit, i=>Y)
    sz = i->put(nbit, i=>Z)
    map(1:(periodic ? nbit : nbit-1)) do i
        j=i%nbit+1
        sx(i)*sx(j)+sy(i)*sy(j)+sz(i)*sz(j)
    end |> sum
end

"""
    transverse_ising(nbit::Int; periodic::Bool=true)

1D transverse ising hamiltonian.
"""
function transverse_ising(nbit::Int; periodic::Bool=true)
    sx = i->put(nbit, i=>X)
    sz = i->put(nbit, i=>Z)
    ising_term = map(1:(periodic ? nbit : nbit-1)) do i
        j=i%nbit+1
        sz(i)*sz(j)
    end |> sum

    ising_term + sum(map(sx, 1:nbit))
end
