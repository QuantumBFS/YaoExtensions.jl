export gatecount

gatecount(blk::AbstractBlock) = gatecount!(blk, Dict{Type{<:AbstractBlock}, Int}())
function gatecount!(c::Union{ChainBlock, KronBlock, PauliString, PutBlock, Bag, Concentrator, Sequence, CachedBlock}, storage::AbstractDict)
    (gatecount!.(c |> subblocks, Ref(storage)); storage)
end

function gatecount!(c::RepeatedBlock, storage::AbstractDict)
    k = typeof(content(c))
    n = length(c.locs)
    if haskey(storage, k)
        storage[k] += n
    else
        storage[k] = n
    end
    storage
end

function gatecount!(c::Union{PrimitiveBlock, Daggered, ControlBlock, ConditionBlock}, storage::AbstractDict)
    k = typeof(c)
    if haskey(storage, k)
        storage[k] += 1
    else
        storage[k] = 1
    end
    storage
end

gatecount!(c::TrivilGate, storage::AbstractDict) = storage
