const Rotor{N, T} = Union{RotationGate{N, T}, PutBlock{N, <:Any, <:RotationGate{<:Any, T}}}
const CphaseGate{N, T} = ControlBlock{N,<:ShiftGate{T},<:Any}
const DiffBlock{N, T} = Union{Rotor{N, T}, CphaseGate{N, T}}

"""
    generator(rot::Rotor) -> AbstractBlock

Return the generator of rotation block.
"""
generator(rot::RotationGate) = rot.block
generator(rot::PutBlock{N, C, GT}) where {N, C, GT<:RotationGate} = PutBlock{N}(generator(rot|>content), rot |> occupied_locs)
generator(c::CphaseGate{N}) where N = ControlBlock{N}(c.ctrl_locs, c.ctrl_config, Z, c.locs)

"""
    apply_back!((ψ, ∂L/∂ψ*), circuit::AbstractBlock, collector) -> AbstractRegister

back propagate and calculate the gradient ∂L/∂θ = 2*Re(∂L/∂ψ*⋅∂ψ*/∂θ), given ∂L/∂ψ*.
`ψ` is the output register, ∂L/∂ψ* should also be register type.

Note: gradients are stored in `Diff` blocks, it can be access by either `diffblock.grad` or `gradient(circuit)`.
Note2: now `apply_back!` returns the inversed gradient!
"""
function apply_back!(st, block::AbstractBlock, collector) #,AbstractContainer{<:PrimitiveBlock}
    out, outδ = st
    if nparameters(block) == 0
        adjblock = block'
        in = apply!(out, adjblock)
        inδ = apply!(outδ, adjblock)
        return (in, inδ)
    else
        throw(MethodError(apply_back, (st, block, collector)))
    end
end

function apply_back!(st, block::Concentrator{N}, collector) where N
    out, outδ = st
    focus!(out, block.locs)
    focus!(outδ, block.locs)
    apply_back!(out, outδ, content(block), collector)
    relax!(out, block.locs; to_nactive=N)
    relax!(outδ, block.locs; to_nactive=N)
    return (out, outδ)
end

function apply_back!(st, block::DiffBlock{N}, collector) where N
    out, outδ = st
    adjblock = block'
    backward_params!((out, outδ), block, collector)
    in = apply!(out, adjblock)
    inδ = apply!(outδ, adjblock)
    return (in, inδ)
end

function apply_back!(st, block::PutBlock{N}, collector) where N
    out, outδ = st
    adjblock = block'
    in = apply!(out, adjblock)
    mat_back!(content(block), mat, collector)
    inδ = apply!(outδ, adjblock)
    return (in, inδ)
end

function apply_back!(st, block::ControlBlock{N}, collector) where N
    out, outδ = st
    adjblock = block'
    in = apply!(out, adjblock)
    mat_back!(content(block), mat, collector)
    inδ = apply!(outδ, adjblock)
    return (in, inδ)
end

function apply_back!(st, block::Daggered, collector)
    error("apply_back fail for type $(typeof(block))")
end

function apply_back!(st, circuit::ChainBlock, collector)
    for blk in Base.Iterators.reverse(subblocks(circuit))
        st = apply_back!(st, blk, collector)
    end
    return st
end

function apply_back!(st, circuit::RepeatedBlock, collector)
end

# TODO: concentrator, repeat, kron
function apply_back!(st, circuit::Concentrator, collector)
    for blk in Base.Iterators.reverse(subblocks(circuit))
        st = apply_back!(st, blk, collector)
    end
    return st
end

apply_back!(st, block::Measure, collector) = throw(MethodError(apply_back!, (st, block, collector)))

function backward_params!(st, block::Rotor, collector)
    in, outδ = st
    Σ = generator(block)
    g = dropdims(sum(conj.(statevec(in |> Σ)) .* statevec(outδ), dims=1), dims=1)
    pushfirst!(collector, -imag(g)/2)
    in |> Σ
    nothing
end
