using YaoExtensions, Test, Yao

@testset "gate count, time" begin
    qc = QFTCircuit(3)
    @test qc |> gatecount |> length == 2
    @test qc |> gatecount |> values |> sum == 6
    qc = rand_supremacy2d(4,4,3)
    gc = qc |> gatecount
    @test gc[HGate] == 32
end
