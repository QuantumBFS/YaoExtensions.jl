using Yao, YaoExtensions, Yao.AD
using Yao.AD: generator
using Test, Random

@testset "BP diff" begin
    Random.seed!(2)
    c = put(4, 3=>Rx(0.5))

    circuit = chain(4, repeat(4, H, 1:4), put(4, 3=>Rz(0.5)), control(4, 2, 1=>shift(0.4)), control(2, 1=>X), put(4, 4=>Ry(0.2)))
    op = heisenberg(4)
    θ = [0.9, 0.2, 0.3]
    dispatch!(circuit, θ)

    for ψ0 in [rand_state(4), rand_state(4, nbatch=10)]
        dispatch!(circuit, θ)
        ψδ, g1 = expect'(op, ψ0 => circuit)
        loss!(θ) = (dispatch!(circuit, θ); expect(op, copy(ψ0) |> circuit))

        g2 = Vector{Any}(undef, length(θ))
        η = 1e-5
        for i in 1:length(θ)
            θ1 = copy(θ)
            θ2 = copy(θ)
            θ1[i] -= 0.5η
            θ2[i] += 0.5η
            g2[i] = (loss!(θ2) - loss!(θ1))/η |> real
        end
        g3 = faithful_grad(op, ψ0 => circuit)
        g2_ = nbatch(ψ0) == 1 ? g2 : dropdims(sum(hcat(g2...); dims=1), dims=1)

        @test isapprox.(g1, g2_, atol=1e-4) |> all
        @test isapprox.(g2, g3, atol=1e-4) |> all
    end
end

@testset "generator" begin
    @test generator(put(4, 1=>Rx(0.1))) == put(4, 1=>X)
    @test generator(Rx(0.1)) == X
    @test generator(cphase(4, 2, 1, 0.1)) == control(4, 2, 1=>Z)
end

@testset "numdiff & opdiff" begin
    @test collect_blocks(XGate, chain([X, Y, Z])) == [X]

    c = chain(put(4, 1=>Rx(0.5)))
    nd = numdiff(c) do cc
        res = expect(put(4, 1=>Z), zero_state(4) |> cc) |> real
        return res
    end

    ed = faithful_grad(put(4, 1=>Z), zero_state(4) => c)
    @test isapprox(nd, ed, atol=1e-4)

    reg = rand_state(4)
    c = chain(put(4, 1=>Rx(0.5)), control(4, 1, 2=>Ry(0.5)), control(4, 1, 2=>shift(0.3)),  kron(4, 2=>Rz(0.3), 3=>Rx(0.7)))
    loss1z(c) = expect(kron(4, 1=>Z, 2=>X), copy(reg) |> c) |> real  # return loss please
    nd = numdiff(loss1z, c)
    ed = faithful_grad(kron(4, 1=>Z, 2=>X), copy(reg) => c)
    @test isapprox(nd, ed, atol=1e-4)

    # the batched version
    reg = rand_state(4, nbatch=10)
    ed2 = faithful_grad(kron(4, 1=>Z, 2=>X), copy(reg) => c)
    nd2 = numdiff(loss1z, c)
    @test isapprox(nd, ed, atol=1e-4)
end

@testset "random diff circuit" begin
    c = variational_circuit(4, 3, [1=>3, 2=>4, 2=>3, 4=>1])
    rots = collect_blocks(RotationGate, c)
    @test length(rots) == nparameters(c) == 40
    @test dispatch!(+, c, ones(40)*0.1) |> parameters == ones(40)*0.1
    @test dispatch!(+, c, :random) |> parameters != ones(40)*0.1

    nbit = 4
    c = variational_circuit(nbit, 1, pair_ring(nbit), mode=:Split)
    reg = rand_state(4)
    dispatch!(c, randn(nparameters(c)))

    dbs = collect_blocks(Diff, c)
    op = kron(4, 1=>Z, 2=>X)
    loss1z(c) = expect(op, copy(reg) |> c)  # return loss please

    # back propagation
    _, bd = expect'(op, reg=>c)

    # get num gradient
    nd = numdiff(loss1z, c)
    ed = faithful_grad(op, copy(reg)=>c)

    @test isapprox.(nd, ed, atol=1e-4) |> all
    @test isapprox.(nd, bd, atol=1e-4) |> all
end
