using Test, Yao, YaoExtensions, Yao.AD

@testset "stat" begin
    nbit = 3
    f(x::Number, y::Number) = Float64(Int64(abs(x-y)) < 1.5)
    x = 0:1<<nbit-1
    h = f.(x', x)
    V = StatFunctional(h)
    VF = StatFunctional{2}(f)
    prs = [1=>2, 2=>3, 3=>1]
    c = ibm_diff_circuit(nbit, 2, prs) |> markdiff
    dispatch!(c, :random)
    dbs = collect_blocks(Diff,c)

    p0 = zero_state(nbit) |> c |> probs
    sample0 = measure(zero_state(nbit) |> c; nshots=5000)
    loss0 = expect(V, p0 |> as_weights)
    gradsn = numdiff.(()->expect(V, zero_state(nbit) |> c |> probs |> as_weights), dbs)
    gradse = statdiff.(()->zero_state(nbit) |> c |> probs |> as_weights, dbs, Ref(V), initial=p0 |> as_weights)
    gradsf = statdiff.(()->measure(zero_state(nbit) |> c; nshots=5000), dbs, Ref(VF), initial=sample0)
    @test all(isapprox.(gradse, gradsn, atol=1e-4))
    @test norm(gradsf-gradse)/norm(gradsf) <= 0.2

    # 1D
    h = randn(1<<nbit)
    V = StatFunctional(h)
    c = ibm_diff_circuit(nbit, 2, prs) |> markdiff
    dispatch!(c, :random)
    dbs = collect_blocks(Diff, c)

    p0 = zero_state(nbit) |> c |> probs |> as_weights
    loss0 = expect(V, p0 |> as_weights)
    gradsn = numdiff.(()->expect(V, zero_state(nbit) |> c |> probs |> as_weights), dbs)
    gradse = statdiff.(()->zero_state(nbit) |> c |> probs |> as_weights, dbs, Ref(V))
    @test all(isapprox.(gradse, gradsn, atol=1e-4))
end

@testset "random diff circuit" begin
    c = variational_circuit(4, 3, [1=>3, 2=>4, 2=>3, 4=>1])
    rots = collect_blocks(RotationGate, c)
    @test length(rots) == nparameters(c) == 40
    @test dispatch!(+, c, ones(40)*0.1) |> parameters == ones(40)*0.1
    @test dispatch!(+, c, :random) |> parameters != ones(40)*0.1

    nbit = 4
    c = variational_circuit(nbit, 1, pair_ring(nbit), mode=:Split) |> markdiff
    reg = rand_state(4)
    dispatch!(c, randn(nparameters(c)))

    dbs = collect_blocks(Diff, c)
    op = kron(4, 1=>Z, 2=>X)
    loss1z() = expect(op, copy(reg) |> c)  # return loss please

    # back propagation
    _, bd = expect'(op, reg=>c)
    @show bd

    # get num gradient
    nd = numdiff.(loss1z, dbs)
    ed = opdiff.(()->copy(reg)|>c, dbs, Ref(op))

    @test isapprox.(nd, ed, atol=1e-4) |> all
    @test isapprox.(nd, bd, atol=1e-4) |> all
end
