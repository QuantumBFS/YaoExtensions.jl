export qft_circuit

"""
    cphase(i, j, k)

Control phase gate.
"""
cphase(i::Int, j::Int, k::Int) = control(i, j=>shift(2π/(1<<k)))
transform(n::Int, i::Int) = chain(n, i==j ? put(i=>H) : cphase(j, i, j-i+1) for j = i:n)

"""
    qft_circuit(n)

Create a Quantum Fourer Transform circuit. See also [`QFT`](@ref).
"""
qft_circuit(n::Int) = chain(n, transform(n, i) for i = 1:n)
