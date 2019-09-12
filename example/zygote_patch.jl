using Zygote: @adjoint

"""
    dispatch_diff_apply!(reg::ArrayReg, circuit::AbstractBlock, diff_params::Vector) -> ArrayReg

Dispatch parameters to differentiable nodes, and evaluate the output.
"""
function dispatch_diff_apply!(reg::ArrayReg, circuit::AbstractBlock, diff_params::Vector)
    dispatch_to_diff!(circuit, diff_params)
    apply!(reg, circuit)
end

@adjoint function dispatch_diff_apply!(reg::ArrayReg{B,T,AT}, circuit::AbstractBlock, diff_params::Vector) where {B,T,AT}
    out = dispatch_diff_apply!(reg, circuit, diff_params)
    saved_out = copy(out)
    out, function (adjout)
        dpsi, dparams = classical_autodiff!(circuit, saved_out, ArrayReg{B,T,AT}(adjout.x.state/2); output_eltype=Any)
        return (dpsi, nothing, dparams)
    end
end

@adjoint Base.copy(reg::ArrayReg) = Base.copy(reg), adjreg->(adjreg,)
