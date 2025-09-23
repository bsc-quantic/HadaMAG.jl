
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

_compute_chunk_mana_SRE(
    istart::Int64,
    iend::Int64,
    ψ::StateVec{ComplexF64,3},
    Zwhere::Vector{Int64},
    XTAB::Matrix{Int64},
) = _compute_chunk_mana_SRE(0, istart, iend, ψ, Zwhere, XTAB)

@fastmath function _compute_chunk_mana_SRE(
    index::Int64,
    istart::Int64,
    iend::Int64,
    ψ::StateVec{ComplexF64,3},
    Zwhere::Vector{Int64},
    XTAB::Matrix{Int64},
)::Tuple{Float64,Float64,ComplexF64}
    L = qudits(ψ)
    dim = 3^L

    p2SAM = m2SAM = m3SAM = 0.0

    TMP = copy(data(ψ))
    Xloc = copy(data(ψ))
    inV = zeros(ComplexF64, dim)
    @assert size(XTAB, 2) == 3^L

    for site = 1:L
        k = XTAB[site, istart]
        k > 0 && actX_qutrit!(TMP, site, k)
    end

    # the worker will update the state TMP when going through the greys code form istart to iend
    for ix = istart:iend
        # non-trivial mathematical thing happening: I need to calculate such a vector related to the
        # state (Xloc) and its propagated version along the grays code, TMP
        for r = 1:dim
            inV[r] = conj(Xloc[r]) * copy(TMP[r])
        end

        # We do fast Hadamard transform of the inVR, inVI
        @inline fast_hadamard_qutrit!(inV)

        # the vectors obtained with FHT  contain overlaps of given Pauli strings:
        # the Pauli strings are of the form XTAB[ix] (0...1 corresponding to Z operator)
        # so to calculate SRE we have to add entries of the resulting vector with the specified powers
        # (depending on the index of SRE we calculate)
        # this step does sum over all Z pauli strings, given their X part determined by XTAB[ix]
        # in time complexity L*2^L (while the naive implementation would be 4^L)

        @inbounds @simd for i in eachindex(inV)
            p = inV[i]
            ap = abs(p)              # |p|
            p2SAM += ap

            m2 = abs2(p)             # |p|^2
            m2SAM += m2 * m2             # |p|^4

            p2 = p * p
            m3SAM += p2 * p             # p^3
        end

        # this takes us from Pauli string at given position of the Greys code to the next one
        # (by a single action of X_j operator )
        if ix + index < dim
            @inline actX_qutrit!(TMP, Zwhere[ix])
        end

    end

    return p2SAM, m2SAM, m3SAM
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