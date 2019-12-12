export QFT, qft
using BitBasis
using LinearAlgebra
using FFTW

"""
    QFT{N} <: PrimitiveBlock{N}

Quantum Fourier Transform node. See also [`qft_circuit`](@ref).
"""
struct QFT{N} <: PrimitiveBlock{N} end

"""
    qft(n)

Create a Quantum Fourier Transform block. See also [`qft_circuit`](@ref).
"""
qft(n) = QFT{n}()

Yao.mat(::Type{T}, q::QFT{N}) where {T, N} = T.(applymatrix(q))

function Yao.apply!(r::ArrayReg, x::QFT)
    α = sqrt(1<<nactive(r))
    st = state(invorder!(r))
    ifft!(st, 1)
    lmul!(α, st)
    return r
end

function Yao.apply!(r::ArrayReg, ::Daggered{<:QFT})
    # (reg.state = invorder_firstdim(fft!(reg|>state, 1)/sqrt(1<<nactive(reg))); reg)
    α = sqrt(1 << nactive(r))
    st = state(r)
    rdiv!(fft!(st, 1), α)
    invorder!(r)
    return r
end

# traits
Yao.ishermitian(q::QFT{N}) where N = N==1
Yao.isreflexive(q::QFT{N}) where N = N==1
Yao.isunitary(q::QFT{N}) where N = true

function Yao.print_block(io::IO, pb::QFT{N}) where N
    printstyled(io, "qft(1-$N)"; bold=true, color=:blue)
end

function Yao.print_block(io::IO, pb::Daggered{N,<:QFT}) where {N, T}
    printstyled(io, "iqft(1-$N)"; bold=true, color=:blue)
end
