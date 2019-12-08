export QFT
using YaoBlocks
using YaoBase
using BitBasis
using YaoArrayRegister
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

mat(::Type{T}, q::QFT{N}) where {T, N} = T.(applymatrix(q))

function apply!(r::ArrayReg{B}, ::QFT) where B
    α = sqrt(1 << nactive(r))
    invorder!(r)
    lmul!(α, ifft!(statevec(r)))
    return r
end

function apply!(r::ArrayReg{B}, ::Daggered{N, <:QFT}) where {B,N}
    α = sqrt(1 << nactive(r))
    invorder!(r)
    lmul!(α, fft!(statevec(r)))
    return r
end

# traits
ishermitian(q::QFT{N}) where N = N==1
isreflexive(q::QFT{N}) where N = N==1
isunitary(q::QFT{N}) where N = true

function print_block(io::IO, pb::QFT{N}) where N
    printstyled(io, "qft(1-$N)"; bold=true, color=:blue)
end

function print_block(io::IO, pb::Daggered{N,<:QFT}) where {N, T}
    printstyled(io, "iqft(1-$N)"; bold=true, color=:blue)
end
