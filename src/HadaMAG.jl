module HadaMAG

include("Helpers.jl")

include("State.jl")
export StateVec,
    load_state, rand_haar, apply_2gate!, apply_2gate, qubits, qudits, qudit_dim, data

include("Backends/Dispatch.jl")

include("SRE2.jl")
export MC_SRE2, SRE2

end
