using Test, Random, LinearAlgebra, LuxurySparse

using Yao
using YaoArrayRegister: invorder
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
    qft = qft_circuit(num_bit)
    iqft = adjoint(qft)
    qftblock = QFT{num_bit}()
    iqftblock = QFT{num_bit}() |> adjoint
    @test openbox(qftblock) == qft
    @test openbox(iqftblock) == iqft
    reg = rand_state(num_bit)

    @test Matrix(mat(chain(3, QFT{3}() |> adjoint, QFT{3}()))) ≈ IMatrix(1<<3)

    # permute lines (Manually)
    @test apply!(copy(reg), iqft) ≈ apply!(copy(reg), QFT{num_bit}() |> adjoint)

    # test fft
    @test apply!(copy(reg), qft) ≈ apply!(copy(reg), qftblock)

    # regression test for nactive
    @test apply!(focus!(copy(reg), 1:3), QFT{3}()) |> isnormalized
end
