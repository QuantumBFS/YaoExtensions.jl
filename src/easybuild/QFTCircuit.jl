export qft_circuit

"""
    cphase(i, j, k)

Control phase gate.
"""
cphase(i::Int, j::Int, k::Int) = control(i, j=>shift(2π/(1<<k)))
hcphases(n::Int, i::Int) = chain(n, i==j ? put(i=>H) : cphase(j, i, j-i+1) for j = i:n)

"""
    qft_circuit(n)

Create a Quantum Fourer Transform circuit. See also [`QFT`](@ref).
"""
qft_circuit(n::Int) = chain(n, hcphases(n, i) for i = 1:n)

qft(l, n) = l == n ? H : chain(
    chain(n, l==j ? put(l=>H) : cphase(j, l, j-l+1) for j = l:n),
    qft(l-1, n)
)

function qft(l, n)
    if l == n
        return put(n, l=>H)
    else
        return chain(
            chain(n, l==j ? put(l=>H) : cphase(j, l, j-l+1) for j = l:n),
            qft(l+1, n)
        )
    end
end
