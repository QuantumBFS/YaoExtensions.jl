using Test
using Yao, BitBasis
using YaoBlocks: ConstGate
using Random
using TupleTools

using SparseArrays, LuxurySparse, LinearAlgebra
include("adjbase.jl")
include("mat_back.jl")
include("apply_back.jl")

ng(f, θ, δ=1e-5) = (f(θ+δ/2) - f(θ-δ/2))/δ
mat_back(T,block,adjm) = mat_back!(T,block,adjm,Float64[])[]
apply_back(st,block) = (col=Float64[]; apply_back!(st,block,col); col[])

function mat_back_jacobian(T, block, θ; use_outeradj=false)
    dispatch!(block, θ)
    m = mat(T, block)
    N = size(m, 1)
    jac = zero(Matrix(m))
    zm = use_outeradj ? OuterProduct(zeros(T,N), zeros(T,N)) : zero(m)
    for j=1:size(m, 2)
        @inbounds for i=1:size(m, 1)
            if m[i,j]!=0
                _setval(zm,i,j,1)
                jac[i,j] = mat_back(ComplexF64, block, zm)
                _setval(zm,i,j,1im)
                jac[i,j] += 1im*mat_back(ComplexF64, block, zm)
                _setval(zm,i,j,0)
            end
        end
    end
    return jac
end
_setval(m::AbstractMatrix, i, j, v) = (m[i,j]=v; m)
_setval(m::OuterProduct, i, j, v) = (m.left[i] = v==0 ? 0 : 1; m.right[j]=v; m)

function apply_back_jacobian(reg0::ArrayReg{B}, block, θ) where B
    dispatch!(block, θ)
    out = apply!(copy(reg0), block)
    m = out.state
    zm = zero(m)
    jac = zero(Matrix(m))
    for j=1:size(m, 2)
        @inbounds for i=1:size(m, 1)
            if m[i,j]!=0
                zm[i,j] = 1
                jac[i,j] = apply_back((copy(out), ArrayReg{B}(copy(zm))), block)
                zm[i,j] *= 1im
                jac[i,j] += 1im*apply_back((copy(out), ArrayReg{B}(copy(zm))), block)
                zm[i,j] = 0
            end
        end
    end
    return jac
end

function test_mat_back(T, block::AbstractBlock{N}, param::Float64; δ=1e-5, use_outeradj::Bool=false) where N
    function mfunc(param)
        dispatch!(block, param)
        mat(T, block)
    end
    # test loss is `real(sum(rand_matrix .* m))`
    got = mat_back_jacobian(T, block, param; use_outeradj=use_outeradj)
    num = ng(mfunc, param, δ)
    res = isapprox(got, num, atol=10*δ)
    if !res
        @show got
        @show num
    end
    return res
end

function test_apply_back(reg0, block::AbstractBlock{N}, param::Float64; δ=1e-5) where N
    function mfunc(param)
        dispatch!(block, param)
        apply!(copy(reg0), block).state
    end
    # test loss is `real(sum(rand_matrix .* m))`
    got = apply_back_jacobian(reg0, block, param)
    num = ng(mfunc, param, δ)
    res = isapprox(got, num, atol=10*δ)
    if !res
        @show got
        @show num
    end
    return res
end

@testset "rot/shift/phase mat grad" begin
    Random.seed!(5)
    for G in [X, Y, Z, ConstGate.SWAP, ConstGate.CZ, ConstGate.CNOT]
        @test test_mat_back(ComplexF64, rot(G, 0.0), 0.5; δ=1e-5)
    end

    for G in [ShiftGate, PhaseGate]
        @test test_mat_back(ComplexF64, G(0.0), 0.5; δ=1e-5)
    end
end

@testset "put block, control block" begin
    Random.seed!(5)
    for use_outeradj in [false, true]
        # put block, diagonal
        @test test_mat_back(ComplexF64, put(3, 1=>Rz(0.5)), 0.5; δ=1e-5, use_outeradj=use_outeradj)
        @test test_mat_back(ComplexF64, control(3, (2,3), 1=>Rz(0.5)), 0.5; δ=1e-5, use_outeradj=use_outeradj)
        # dense matrix
        @test test_mat_back(ComplexF64, put(3, 1=>Rx(0.5)), 0.5; δ=1e-5, use_outeradj=use_outeradj)
        @test test_mat_back(ComplexF64, control(3, (2,3), 1=>Rx(0.5)), 0.5; δ=1e-5, use_outeradj=use_outeradj)
        # sparse matrix csc
        @test test_mat_back(ComplexF64, put(3, (1,2)=>rot(SWAP, 0.5)), 0.5; δ=1e-5, use_outeradj=use_outeradj)
        @test test_mat_back(ComplexF64, control(3, (3,), (1,2)=>rot(SWAP, 0.5)), 0.5; δ=1e-5, use_outeradj=use_outeradj)
    end

    # is permatrix even possible?
    #@test test_mat_back(ComplexF64, put(3, 1=>matblock(pmrand(2))), [0.5, 0.6]; δ=1e-5)
    # ignore identity matrix.
end

@testset "apply put" begin
    Random.seed!(5)
    reg0 = rand_state(3)
    # put block, diagonal
    @test test_apply_back(reg0, put(3, 1=>Rz(0.5)), 0.5; δ=1e-5)
    @test test_apply_back(reg0, control(3, (2,3), 1=>Rz(0.5)), 0.5; δ=1e-5)
    # dense matrix
    @test test_apply_back(reg0, put(3, 1=>Rx(0.5)), 0.5; δ=1e-5)
    @test test_apply_back(reg0, control(3, (2,3), 1=>Rx(0.5)), 0.5; δ=1e-5)
    # sparse matrix csc
    @test test_apply_back(reg0, put(3, (1,2)=>rot(SWAP, 0.5)), 0.5; δ=1e-5)
    @test test_apply_back(reg0, control(3, (3,), (1,2)=>rot(SWAP, 0.5)), 0.5; δ=1e-5)
end

@testset "chain mat grad" begin
    nbit = 5
    θ = 0.5
    # Chain Block
    Random.seed!(5)
    b = randn(ComplexF64, 2)
    gctrl(x) = (b'*mat(chain([Rx(x),Ry(x+0.4)]))*b)[] |> real   # collect not correctly defined
    @test isapprox(gctrl'(θ), ng(gctrl, θ), atol=1e-4)
end

@testset "general mat grad" begin
    v = randn(ComplexF64, 4)
    circuit = chain(2, [put(2, 2=>Rx(0.0)), control(2, 1, 2=>Z), put(2, 2=>Rz(0.0))])
    circuit = dispatch!(circuit, [0.4, 0.6])
    function l1(circuit)
        (v'* mat(circuit) * v)[] |> real
    end
    g1 = x->(dispatch!(circuit, [x, 0.6]); l1(circuit))
    g2 = x->(dispatch!(circuit, [0.4, x]); l1(circuit))
    @test collect_gradients(l1'(circuit)) ≈ [ng(g1, 0.4), ng(g2, 0.6)]
    function loss1(params)
        dispatch!(circuit, params)   # dispatch! will fail for deep models!!!!!
        (v'* mat(circuit) * v)[] |> real
    end
    @show loss1'([0.4, 0.5])
    @test_broken gradient_check(loss1, [0.4, 0.6])
end

@testset "kron mat grad" begin
    Random.seed!(2)
    nbit = 2
    θ = 1.2
    # Kron Block
    b = randn(ComplexF64, 1<<nbit)
    gkron(x) = (b'*mat(kron(nbit, 1=>Rx(x), 2=>Rz(x+0.8)))*b)[] |> real
    @test isapprox(gkron'(θ), ng(gkron, θ), atol=1e-5)

    nbit = 5
    b = randn(ComplexF64, 1<<nbit)
    gkron2(x) = (b'*mat(kron(nbit, 4=>Rx(x), 1=>Rz(x+0.8)))*b)[] |> real
    @test isapprox(gkron2'(θ), ng(gkron2, θ), atol=1e-5)

    nbit = 5
    b = randn(ComplexF64, 1<<nbit)
    gkron3(x) = (b'*mat(kron(nbit, 4=>Rx(x), 1=>Ry(x+0.4)))*b)[] |> real
    @test isapprox(gkron3'(θ), ng(gkron3, θ), atol=1e-5)
end
