using Test
using Yao, BitBasis
using YaoBlocks: ConstGate
using Random
using TupleTools

using SparseArrays, LuxurySparse, LinearAlgebra
include("adjbase.jl")
include("adjroutines.jl")
include("mat_back.jl")
include("apply_back.jl")
include("gradcheck.jl")

@testset "mat rot/shift/phase" begin
    Random.seed!(5)
    for G in [X, Y, Z, ConstGate.SWAP, ConstGate.CZ, ConstGate.CNOT]
        @test test_mat_back(ComplexF64, rot(G, 0.0), 0.5; δ=1e-5)
    end

    for G in [ShiftGate, PhaseGate]
        @test test_mat_back(ComplexF64, G(0.0), 0.5; δ=1e-5)
    end
end

@testset "mat put block, control block" begin
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
    for reg0 in [rand_state(3), rand_state(3, nbatch=10)]
        # put block, diagonal
        @test test_apply_back(reg0, put(3, 1=>shift(0.5)), 0.5; δ=1e-5)
        @test test_apply_back(reg0, control(3, (2,3), 1=>Rz(0.5)), 0.5; δ=1e-5)
        @test test_apply_back(reg0, control(3, 2, 1=>shift(0.5)), 0.5; δ=1e-5)
        # dense matrix
        @test test_apply_back(reg0, put(3, 1=>cache(Rx(0.5))), 0.5; δ=1e-5)
        @test test_apply_back(reg0, control(3, (2,3), 1=>Rx(0.5)), 0.5; δ=1e-5)
        # sparse matrix csc
        @test test_apply_back(reg0, put(3, (1,2)=>rot(SWAP, 0.5)), 0.5; δ=1e-5)
        @test test_apply_back(reg0, control(3, (3,), (1,2)=>rot(SWAP, 0.5)), 0.5; δ=1e-5)

        # special cases: DiffBlock
        @test test_apply_back(reg0, put(3, 1=>Rz(0.5)), 0.5; δ=1e-5)
        @test test_apply_back(reg0, put(3, 1=>Rx(0.5)), 0.5; δ=1e-5)
        @test test_apply_back(rand_state(1), Rx(0.0), 0.5; δ=1e-5)
    end
end

@testset "apply chain" begin
    Random.seed!(5)
    for reg0 in [rand_state(3), rand_state(3, nbatch=10)]
        @test test_apply_back(reg0, chain(3, put(3, 1=>shift(0.0)), control(3, (2,3), 1=>Rz(0.0))), [0.5,0.5]; δ=1e-5)
    end
end

@testset "apply dagger, scale" begin
    Random.seed!(5)
    for reg0 in [rand_state(3), rand_state(3, nbatch=10)]
        @test test_apply_back(reg0, chain(put(3, 1=>Rx(0.0)), (3+2im)*control(3, (2,3), 1=>Rz(0.0))), [0.5,0.5]; δ=1e-5)
        @test test_apply_back(reg0, Daggered(put(3, 1=>Rx(0.0))), 0.5; δ=1e-5)
        @test test_apply_back(reg0, control(3, (2,3), 1=>Daggered(Rz(0.0))), 0.5; δ=1e-5)
        @test test_apply_back(reg0, chain(3, Daggered(put(3, 1=>Rx(0.0))), control(3, (2,3), 1=>Daggered(Rz(0.0)))), [0.5,0.5]; δ=1e-5)
    end
end

@testset "apply concentrate" begin
    Random.seed!(5)
    for reg0 in [rand_state(3), rand_state(3, nbatch=10)]
        @test test_apply_back(reg0, chain(3,  put(3, 1=>Rx(0.0)), concentrate(3, control(2, 2,1=>shift(0.0)), (3,1))), [0.5,0.5]; δ=1e-5)
    end
end

@testset "mat concentrate" begin
    Random.seed!(5)
    @test test_mat_back(ComplexF64, concentrate(3, control(2, 2,1=>shift(0.0)), (3,1)), 0.5; δ=1e-5)
end

@testset "apply kron" begin
    Random.seed!(5)
    for reg0 in [rand_state(3), rand_state(3, nbatch=10)]
        @test test_apply_back(reg0, chain(3, put(3, 1=>Rx(0.0)), kron(Rx(0.4), Y, Rz(0.5))), [0.5,0.3, 0.8]; δ=1e-5)
    end
end

@testset "mat chain" begin
    Random.seed!(5)
    @test test_mat_back(ComplexF64, chain(3, control(3, 2,1=>shift(0.0))), 0.5; δ=1e-5)
    @test test_mat_back(ComplexF64, chain(3, put(3, 2=>X), control(3, 2,1=>shift(0.0))), 0.5; δ=1e-5)
    @test test_mat_back(ComplexF64, chain(3, control(3, 2,1=>shift(0.0)), put(3, 2=>X)), 0.5; δ=1e-5)
    @test test_mat_back(ComplexF64, chain(3, control(3, 2,1=>shift(0.0)), put(3, 1=>Rx(0.0))), [0.5,0.5]; δ=1e-5)
end

@testset "mat Add" begin
    @test test_mat_back(ComplexF64, (3*control(3, (2,3), 1=>Rz(0.0))+put(3,2=>Rx(0.0)))/5, [0.5,0.5]; δ=1e-5)
end

@testset "mat kron" begin
    use_outeradj=false
    @test_broken test_mat_back(ComplexF64, kron(Rx(0.5), Rz(0.6)), [0.5, 0.5]; δ=1e-5, use_outeradj=use_outeradj)
end

@testset "apply Add" begin
    # + is not reversible
    @test_broken test_apply_back(reg0, (3*control(3, (2,3), 1=>Rz(0.0))+put(3,2=>Rx(0.0)))/5, [0.5,0.5]; δ=1e-5)
end
