import Yao: gatecount

function gatecount!(c::Union{PauliString, Bag, Sequence}, storage::AbstractDict)
    (gatecount!.(c |> subblocks, Ref(storage)); storage)
end

function gatecount!(c::ConditionBlock, storage::AbstractDict)
    k = typeof(c)
    if haskey(storage, k)
        storage[k] += 1
    else
        storage[k] = 1
    end
    storage
end

gatecount!(c::TrivialGate, storage::AbstractDict) = storage
