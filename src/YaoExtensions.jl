module YaoExtensions

using LuxurySparse
using Yao, YaoBlocks.ConstGate, BitBasis

include("Miscellaneous.jl")

# new block types
include("block_extension/blocks.jl")

# easy build
include("easybuild/easybuild.jl")

# AD
include("diff_extension/diffs.jl")

include("gatecount.jl")

include("deprecate.jl")

end # module
