
"""
    Mana(ψ::StateVec{T,3}; backend = :auto)

Compute the Mana of a quantum qutrit state `ψ` using the HadaMAG algorithm.
Returns the Mana value.

# Arguments
- `ψ`: A [`StateVec`](@ref) object representing the quantum state.

# Keyword Arguments
- `backend`: The backend to use for the computation. Default is `:auto`, which selects the best available backend.
- `progress`: Whether to show a progress bar. Default to `true`.
"""
function Mana(ψ::StateVec{T,3}; backend = :auto, progress = true) where {T}
    _apply_backend(_choose_backend(backend), :Mana, ψ)
end

@fastmath function compute_chunk_mana_qutrits(
    istart::Int,
    iend::Int,
    ψ::StateVec{ComplexF64,3},
    Zwhere::Vector{Int},
    XTAB::Matrix{Int},
    TMP::Vector{ComplexF64},
    conj_Xloc::Vector{ComplexF64},
    inV::Vector{ComplexF64},
    pbar::AbstractProgress,
    stride::Int,
)::Float64
    L = qudits(ψ)
    dim = 3^L
    cnt = 0

    # Build initial permutation for column istart (apply each site's k)
    perm = collect(1:dim)
    @inbounds for s in 1:L
        k = XTAB[s, istart] % 3
        if k == 1
            rotate_perm_site!(perm, s, 1)
        elseif k == 2
            rotate_perm_site!(perm, s, 2)
        end
    end

    p2SAM = 0.0
    @inbounds for ix = istart:iend
        # cheap progress tick every `stride`
        if stride > 0
            cnt += 1
            if (cnt % stride) == 0
                tick!(pbar, stride)
                cnt = 0
            end
        end

        @simd for i in 1:dim
            @inbounds inV[i] = conj_Xloc[i] * TMP[perm[i]]
        end

        fast_hadamard_qutrit!(inV)

        @simd for i in 1:dim
            p  = inV[i]
            ap = abs(p)
            p2SAM += ap
        end

        # Advance permutation to the next column (single-site update)
        if ix < iend
            s = Zwhere[ix] # 1-based site whose digit changes
            rotate_perm_site!(perm, s, 1)   # or 2, if your schedule steps by 2
        end
    end

    finish!(pbar)

    return p2SAM
end

# Rotate the permutation vector `perm` in-place to reflect applying
# the qutrit X-gate at `site` (1-based) `k` times (k=1 or 2).
# This function modifies `perm` directly.
@inline function rotate_perm_site!(perm::Vector{Int}, site::Int, k::Int)
    @assert k==1 || k==2
    stride = Int(3)^(site-1)
    block  = 3*stride
    @inbounds for bs = 1:block:length(perm)
        @simd for off = 0:stride-1
            i0 = bs + off
            i1 = i0 + stride
            i2 = i1 + stride
            if k == 1
                perm[i0], perm[i1], perm[i2] = perm[i1], perm[i2], perm[i0]   # 0→1→2
            else
                perm[i0], perm[i1], perm[i2] = perm[i2], perm[i0], perm[i1]   # 0→2→1
            end
        end
    end
end