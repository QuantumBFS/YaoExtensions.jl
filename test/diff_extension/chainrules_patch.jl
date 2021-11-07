import Zygote, ForwardDiff
using YaoExtensions, Random, Test
using Yao

@testset "adwith zygote" begin
    c = YaoExtensions.variational_circuit(5)
    dispatch!(c, :random)

    function loss(reg::AbstractRegister, circuit::AbstractBlock{N}) where N
        #copy(reg) |> circuit
        reg = apply(copy(reg), circuit)
        st = state(reg)
        sum(real(st.*st))
    end

    reg0 = zero_state(5)
    params = rand!(parameters(c))
    paramsδ = Zygote.gradient(params->loss(reg0, dispatch(c, params)), params)[1]
    regδ = Zygote.gradient(reg->loss(reg, c), reg0)[1]
    fparamsδ = ForwardDiff.gradient(params->loss(ArrayReg(Matrix{Complex{eltype(params)}}(reg0.state)), dispatch(c, params)), params)
    fregδ = ForwardDiff.gradient(x->loss(ArrayReg([Complex(x[2i-1],x[2i]) for i=1:length(x)÷2]), dispatch(c, Vector{real(eltype(x))}(parameters(c)))), reinterpret(Float64,reg0.state))
    @test fregδ ≈ reinterpret(Float64, regδ.state)
    @test fparamsδ ≈ paramsδ
end