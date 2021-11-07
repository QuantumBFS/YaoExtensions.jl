@testset "faithful opdiff" begin
    include("faithful_opdiff.jl")
end

@testset "stats diff" begin
    include("stats.jl")
end

@testset "chainrules_patch" begin
    include("chainrules_patch.jl")
end
