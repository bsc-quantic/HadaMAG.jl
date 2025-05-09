module HadaMAG

# Write your package code here.
export func

"""
    func(x)

Return double the number `x` plus `1`.
"""
func(x) = 2x + 1

include("Helpers.jl")

include("State.jl")
export StateVec, load_state, rand_haar

end
