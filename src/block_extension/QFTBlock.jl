export QFTBlock

struct QFTBlock{N} <: PrimitiveBlock{N} end
mat(::Type{T}, q::QFTBlock{N}) where {T, N} = T.(applymatrix(q))

apply!(reg::DefaultRegister{B}, ::QFTBlock) where B = (reg.state = ifft!(invorder_firstdim(reg |> state), 1)*sqrt(1<<nactive(reg)); reg)
apply!(reg::DefaultRegister{B}, ::Daggered{N, <:QFTBlock}) where {B,N} = (reg.state = invorder_firstdim(fft!(reg|>state, 1)/sqrt(1<<nactive(reg))); reg)

# traits
ishermitian(q::QFTBlock{N}) where N = N==1
isreflexive(q::QFTBlock{N}) where N = N==1
isunitary(q::QFTBlock{N}) where N = true

function print_block(io::IO, pb::QFTBlock{N}) where N
    printstyled(io, "QFT(1-$N)"; bold=true, color=:blue)
end

function print_block(io::IO, pb::Daggered{N,<:QFTBlock}) where {N, T}
    printstyled(io, "IQFT(1-$N)"; bold=true, color=:blue)
end
