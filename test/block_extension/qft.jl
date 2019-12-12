using Test, Random, LinearAlgebra, LuxurySparse

using Yao
using Yao.YaoArrayRegister: invorder
using YaoExtensions
using FFTW

@testset "QFT" begin
    num_bit = 5
    fftblock = qft_circuit(num_bit)
    ifftblock = fftblock'
    reg = rand_state(num_bit)
    rv = copy(statevec(reg))

    @test Matrix(mat(chain(3, qft_circuit(3) |> adjoint, qft_circuit(3)))) ≈ IMatrix(1<<3)

    # test ifft
    reg1 = apply!(copy(reg), ifftblock)

    # permute lines (Manually)
    kv = fft(statevec(reg))/sqrt(length(rv))
    @test statevec(reg1) ≈ invorder(kv)

    # test fft
    reg2 = apply!(invorder!(copy(reg)), fftblock)
    kv = ifft(rv) * sqrt(length(rv))
    @test statevec(reg2) ≈ kv
end


@testset "QFT" begin
    num_bit = 5
    circuit_qft = qft_circuit(num_bit)
    circuit_iqft = circuit_qft'
    block_qft = qft(num_bit)
    block_iqft = qft(num_bit)'
    @test openbox(block_qft) == circuit_qft
    @test openbox(block_iqft) == circuit_iqft
    reg = rand_state(num_bit)

    @test Matrix(chain(3, qft(3)', qft(3))) ≈ IMatrix(1<<3)

    # permute lines (Manually)
    @test apply!(copy(reg), circuit_iqft) ≈ apply!(copy(reg), block_iqft)

    # test fft
    @test apply!(copy(reg), qft) ≈ apply!(copy(reg), block_qft)

    # regression test for nactive
    @test apply!(focus!(copy(reg), 1:3), QFT{3}()) |> isnormalized
end
