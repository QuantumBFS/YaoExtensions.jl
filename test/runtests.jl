using Test, Random, LinearAlgebra
using Yao
using YaoExtensions

@testset "block extension" begin
    include("block_extension/blocks.jl")
end

@testset "easybuild" begin
    include("easybuild/easybuild.jl")
end

@testset "diff_extension" begin
    include("diff_extension/diffs.jl")
end

@testset "gatecount" begin
    include("gatecount.jl")
end
