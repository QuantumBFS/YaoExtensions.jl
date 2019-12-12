@deprecate autodiff(mode, block) markdiff(block)
@deprecate autodiff(mode) markdiff
@deprecate opdiff faithful_grad
@deprecate statdiff faithful_grad
@deprecate QFTCircuit(n) qft_circuit(n)

struct QFTBlock{N} <: PrimitiveBlock{N}
    @deprecate QFTBlock{n}() where n qft(n)
end
