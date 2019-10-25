using Test, Yao, YaoExtensions, Yao.AD
using BitBasis, Random

@testset "kernels" begin
    @test rbf_kernel(0.5)(3, 4) == rbf_kernel(0.5)(BitStr64{5}(3), BitStr64{5}(4))
    @test brbf_kernel(0.5)(3, 4) == brbf_kernel(0.5)(BitStr64{5}(3), BitStr64{5}(4))
    for sf in [rbf_functional(0.5), StatFunctional{1}(sin∘bfloat)]
        wv = YaoExtensions.witness_vec(sf, randn(4))
        @test ndims(wv) == 1
        wv = YaoExtensions.witness_vec(sf, randn(4, 10))
        @test ndims(wv) == 2
    end
end

@testset "stats basic 2d" begin
    Random.seed!(2)
    for V in [rbf_functional(0.5), StatFunctional{1}(sin∘bfloat)]
        res = expect(V, zero_state(2) |> put(2,1=>H) |> as_weights)
        @test res isa Float64
        res2 = expect(V, measure(zero_state(2) |> put(2,1=>H); nshots=1000))
        @test res2 isa Float64
        @test isapprox(res, res2, atol=1e-2)

        res3 = expect(V, zero_state(2, nbatch=10) |> put(2,1=>H) |> as_weights)
        @test length(res3) == 10 && all(res3 .≈ res)

        res4 = expect(V, measure(zero_state(2, nbatch=10) |> put(2,1=>H); nshots=1000))
        @test length(res3) == 10 && all(isapprox.(res3, res, atol=1e-2))
    end
end

@testset "stat diff 2d" begin
    Random.seed!(2)
    nbit = 3
    # 2D
    x = 0:1<<nbit-1
    V = rbf_functional(0.5)
    c = variational_circuit(nbit)
    dispatch!(c, :random)
    dbs = collect_blocks(Diff,c)

    for nbatch=[1, 10]
        p0 = zero_state(nbit; nbatch=nbatch) |> c |> probs
        sample0 = measure(zero_state(nbit) |> c; nshots=5000)
        loss0 = expect(V, p0 |> as_weights)
        gradsn = numdiff(c->expect(V, zero_state(nbit; nbatch=nbatch) |> c |> as_weights), c)
        gradse = faithful_grad(V, zero_state(nbit; nbatch=nbatch) => c)
        gradsa = expect'(V, zero_state(nbit; nbatch=nbatch) => c)[2]
        @test isapprox(gradse, gradsn, atol=1e-4)
        @test isapprox(sum.(gradse), gradsa, atol=1e-4)
    end
end

@testset "stat diff 1d" begin
    Random.seed!(2)
    nbit = 3
    # 1D
    V = StatFunctional{1}(sin ∘ bfloat)
    c = variational_circuit(nbit)
    dispatch!(c, :random)

    for nbatch=[1, 100]
        p0 = zero_state(nbit; nbatch=nbatch) |> c |> probs
        loss0 = expect(V, p0 |> as_weights)
        gradsn = numdiff(c->expect(V, zero_state(nbit; nbatch=nbatch) |> c |> as_weights), c)
        gradse = faithful_grad(V, zero_state(nbit; nbatch=nbatch) => c)
        gradsa = expect'(V, zero_state(nbit; nbatch=nbatch) => c)[2]
        @test isapprox(gradse, gradsn, atol=1e-4)
        @test isapprox(sum.(gradse), gradsa, atol=1e-4)
    end
end

@testset "mmd loss" begin
    Random.seed!(4)
    nbit = 3
    # 2D
    x = 0:1<<nbit-1
    V = rbf_mmd_loss(0.5, normalize!(rand(1<<nbit)))
    c = variational_circuit(nbit)
    dispatch!(c, :random)
    dbs = collect_blocks(Diff,c)

    for nbatch=[1, 10]
        p0 = zero_state(nbit; nbatch=nbatch) |> c |> probs
        sample0 = measure(zero_state(nbit) |> c; nshots=5000)
        loss0 = expect(V, p0 |> as_weights)
        gradsn = numdiff(c->expect(V, zero_state(nbit; nbatch=nbatch) |> c |> as_weights), c)
        gradse = faithful_grad(V, zero_state(nbit; nbatch=nbatch) => c)
        gradsa = expect'(V, zero_state(nbit; nbatch=nbatch) => c)[2]
        @test isapprox(gradse, gradsn, atol=1e-3)
        @test isapprox(sum.(gradse), gradsa, atol=1e-3)
    end
end
