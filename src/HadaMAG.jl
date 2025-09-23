module HadaMAG

include("Helpers.jl")

include("Progress.jl")

include("Basis.jl")
export generate_gray_table

include("State.jl")
export StateVec,
    load_state, rand_haar, apply_2gate!, apply_2gate, qubits, qudits, qudit_dim, data

include("Backends/Dispatch.jl")

include("SRE.jl")
export MC_SRE, SRE

include("Mana.jl")
export Mana

end
