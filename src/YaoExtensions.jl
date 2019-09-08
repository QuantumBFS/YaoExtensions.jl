module YaoExtensions

using LuxurySparse
using Yao, YaoBlocks.ConstGate, BitBasis
using SymEngine
import Yao: mat, dispatch!, niparams, getiparams, setiparams!,
        cache_key, print_block, apply!, PrimitiveBlock, ishermitian, isunitary, isreflexive
import YaoBlocks: render_params, chsubblocks, subblocks
import Base: ==, copy, hash

include("Miscellaneous.jl")

# new block types
include("sequence.jl")
include("Diff.jl")
include("TrivilGate.jl")
include("Bag.jl")
include("Mod.jl")
include("ConditionBlock.jl")
include("RotBasis.jl")
include("reflect_gate.jl")
include("math_gate.jl")
include("pauli_strings.jl")

# easy build
include("QFT.jl")
include("CircuitBuild.jl")
include("supremacy_circuit.jl")
include("hamiltonians.jl")

include("timer.jl")

end # module
