using StatsBase
using Yao.AD
using Yao.AD: Rotor

############# manipulate quantum differential node ############
Yao.AD.generator(c::CPhaseGate{N}) where N = ControlBlock{N}(c.ctrl_locs, c.ctrl_config, Z, c.locs)

function get_diffblocks(circuit::AbstractBlock)
    diffblocks = collect_blocks(Union{RotationGate, CPhaseGate}, circuit)
    length(diffblocks) == nparameters(circuit) || throw(ArgumentError("Input circuit contains non-differentiable/unsupported parameters."))
    return diffblocks
end

#### interface #####
export numdiff, faithful_grad

@inline function _perturb(func, gate::AbstractBlock, δ::Real)
    dispatch!(-, gate, (δ,))
    r1 = func()
    dispatch!(+, gate, (2δ,))
    r2 = func()
    dispatch!(-, gate, (δ,))
    r1, r2
end

"""
    numdiff(loss, circuit::AbstractBlock, δ::Real=1e-2) => Vector

Numeric differentiation a loss over a circuit, the loss take the circuit as input.
"""
@inline function numdiff(loss, circuit::AbstractBlock, δ::Real=1e-2)
    map(get_diffblocks(circuit)) do diffblock
        r1, r2 = _perturb(()->loss(circuit), diffblock, δ)
        (r2 - r1)/2δ
    end
end

"""
    faithful_grad(x, pair::Pair{<:ArrayReg, <:AbstractBlock}) -> Vector

Differentiate `x` over all parameters. `x` can be an `AbstractBlock`, `StatFunctional` or `MMD`.
"""
@inline function faithful_grad(op::AbstractBlock, pair::Pair{<:ArrayReg, <:AbstractBlock})
    map(get_diffblocks(pair.second)) do diffblock
        r1, r2 = _perturb(()->expect(op, copy(pair.first) |> pair.second) |> real, diffblock, π/2)
        (r2 - r1)/2
    end
end
