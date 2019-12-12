export openbox

"""
    openbox(block::AbstractBlock) -> AbstractBlock

For a black box, like QFTBlock, you can get its white box (faithful simulation) using this function.
"""
openbox(q::QFT{N}) where N = qft_circuit(N)
openbox(q::Daggered{<:QFT, N}) where {N} = qft_circuit(N)'

function openbox(fs::FSimGate)
    if fs.theta ≈ π/2
        return cphase(2,2,1,-fs.phi)*SWAP*rot(kron(Z,Z), -π/2)*put(2,1=>phase(-π/4))
    else
        return cphase(2,2,1,-fs.phi)*rot(SWAP,2*fs.theta)*rot(kron(Z,Z), -fs.theta)*put(2,1=>phase(fs.theta/2))
    end
end
