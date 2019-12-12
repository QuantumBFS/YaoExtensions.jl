module YaoExtensions

using LuxurySparse
using BitBasis
using Yao
using Yao: YaoBlocks, YaoArrayRegister
using .YaoBlocks.ConstGate

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
