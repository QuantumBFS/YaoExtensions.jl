entangler_google53(nbits::Int, i::Int, j::Int) = put(nbits, (i,j)=>FSimGate(π/2, π/6))
