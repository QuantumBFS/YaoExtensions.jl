using Test
@testset "variational_circuit" begin
    include("variational_circuit.jl")
end

@testset "hamiltonians" begin
    include("hamiltonians.jl")
end

@testset "supremacy_circuit" begin
    include("supremacy_circuit.jl")
end

@testset "google53" begin
    include("google53.jl")
end
