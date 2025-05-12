module HadaMAG

include("Helpers.jl")

include("State.jl")
export StateVec, load_state, rand_haar, apply_2gate!, apply_2gate, qubits, qudits, qudit_dim

end
