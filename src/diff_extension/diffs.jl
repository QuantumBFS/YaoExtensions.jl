_dropdims(v::Vector; dims) = length(v) == 1 ? v[] : dropdims(v; dims=dims)
_dropdims(a; dims) = dropdims(a; dims=dims)
include("chainrules_patch.jl")
include("faithful_opdiff.jl")
include("stats.jl")
include("kernel_mmd.jl")
