using Test, Yao, YaoExtensions, Yao.AD
using BitBasis

@testset "stat" begin
    nbit = 3
    # 2D
    f(x::Number, y::Number) = Float64(Int64(abs(x-y)) < 1.5)
    x = 0:1<<nbit-1
    V = StatFunctional{2}(f)
    c = variational_circuit(nbit) |> markdiff
    dispatch!(c, :random)
    dbs = collect_blocks(Diff,c)

    p0 = zero_state(nbit) |> c |> probs
    sample0 = measure(zero_state(nbit) |> c; nshots=5000)
    loss0 = expect(V, p0 |> as_weights)
    gradsn = numdiff(c->expect(V, zero_state(nbit) |> c |> probs |> as_weights), c)
    gradse = statdiff(V, zero_state(nbit) => c)
    gradsa = expect'(V, zero_state(nbit) => c)
    @test all(isapprox.(gradse, gradsn, atol=1e-4))
    @test all(isapprox.(gradse, gradsa, atol=1e-4))

    # 1D
    V = StatFunctional{1}(sin ∘ bfloat)
    c = variational_circuit(nbit) |> markdiff
    dispatch!(c, :random)

    p0 = zero_state(nbit) |> c |> probs
    loss0 = expect(V, p0 |> as_weights)
    gradsn = numdiff(c->expect(V, zero_state(nbit) |> c |> probs |> as_weights), c)
    gradse = statdiff(V, zero_state(nbit) => c)
    gradsa = expect'(V, zero_state(nbit) => c)
    @test all(isapprox.(gradse, gradsn, atol=1e-4))
    @test all(isapprox.(gradse, gradsa, atol=1e-4))
end
