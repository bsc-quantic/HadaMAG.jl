
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
    _apply_backend(_choose_backend(backend), :Mana, ψ; progress)
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

"""
    build_single_qutrit_M()

Return `(M, A_list)` where:
- `A_list[idx]` is the 3×3 phase-point operator A_u (u ≡ idx in 0..8),
- `M` is 9×9 such that for one qutrit: `w = M * vec(ρ)` (column-major vec)
    and `w[idx] = tr(A_list[idx] * ρ)`.

This matches the C++ construction exactly (including the use of vec(A_u^T) as rows of M).
"""
function build_single_qutrit_M()
    d = 3
    num_ops = 9

    ω = cis(2π / 3)  # exp(i 2π/3)

    # Shift X
    X = zeros(ComplexF64, d, d)
    for k in 0:d-1
        X[(k + 1) % d + 1, k + 1] = 1.0 + 0im
    end

    # Clock Z
    Z = zeros(ComplexF64, d, d)
    for k in 0:d-1
        Z[k + 1, k + 1] = ω^k
    end

    # Precompute powers: I, A, A^2
    I3 = Matrix{ComplexF64}(I, d, d)
    Zpow = (I3, Z, Z * Z)
    Xpow = (I3, X, X * X)

    two_inv = 2  # inverse of 2 mod 3

    # T_{a,ap}
    T_list = [zeros(ComplexF64, d, d) for _ in 1:num_ops]
    for a in 0:d-1
        for ap in 0:d-1
            idx = a * 3 + ap  # 0..8
            exponent = mod(-two_inv * a * ap, d)  # 0..2
            phase = ω^exponent
            T_list[idx + 1] .= phase .* (Zpow[a + 1] * Xpow[ap + 1])
        end
    end

    # A0 = (1/d) * sum_u T_u
    A0 = zeros(ComplexF64, d, d)
    for Tu in T_list
        A0 .+= Tu
    end
    A0 ./= d

    # A_u = T_u A0 T_u^†
    A_list = [zeros(ComplexF64, d, d) for _ in 1:num_ops]
    for idx in 1:num_ops
        Tu = T_list[idx]
        A_list[idx] .= Tu * A0 * Tu'   # ' is adjoint in Julia
    end

    # Build M (9×9): row idx is vec_col(A_u^T)
    M = zeros(ComplexF64, num_ops, num_ops)
    for idx in 1:num_ops
        Au = A_list[idx]
        # vec(transpose(Au)) is column-major vec of Au^T (NO conjugation)
        M[idx, :] .= vec(transpose(Au))
    end

    return M, A_list
end

# ---------- Tensor-product-compatible vec for N qutrits ----------
"""
    vec_rho_tensor_N(ρ::AbstractMatrix, N::Int)

Map a 3^N×3^N density matrix `ρ` (column-major) to a vector `v` of length 9^N
such that each site contributes an index idx9 = 3*i_k + j_k.

This matches the C++ digit convention: the vector legs are ordered MSB→LSB in base-3.
"""
function vec_rho_tensor_N(ρ::AbstractMatrix{Tc}, N::Int) where {Tc<:Complex}
    d = 3
    dim = ipow(d, N)
    size(ρ, 1) == dim && size(ρ, 2) == dim ||
        throw(ArgumentError("ρ must be $(dim)×$(dim) for N=$N qutrits"))

    op_dim = ipow(d * d, N)  # 9^N
    v = zeros(Tc, op_dim)

    i_digits = Vector{Int}(undef, N)
    j_digits = Vector{Int}(undef, N)

    @inbounds for c0 in 0:dim-1
        tmpc = c0
        for k in N:-1:1
            j_digits[k] = tmpc % d
            tmpc ÷= d
        end

        for r0 in 0:dim-1
            tmpr = r0
            for k in N:-1:1
                i_digits[k] = tmpr % d
                tmpr ÷= d
            end

            p = 0
            for k in 1:N
                idx9 = i_digits[k] * 3 + j_digits[k]  # 0..8
                p = p * 9 + idx9
            end

            # Julia is 1-based; ρ is a Matrix so ρ[r0+1, c0+1] matches column-major storage
            v[p + 1] = ρ[r0 + 1, c0 + 1]
        end
    end

    return v
end

"""
    apply_M_tensor_N_inplace!(v, M, N)

In-place apply `M^{⊗N}` to `v` (length 9^N), sweeping legs where each leg has dimension 9.
Leg 1 corresponds to the most significant base-9 digit.
"""
function apply_M_tensor_N_inplace!(v::AbstractVector{Tc}, M::AbstractMatrix{Tc}, N::Int) where {Tc<:Complex}
    local_dim = 9
    total_dim = length(v)
    total_dim == ipow(local_dim, N) || throw(ArgumentError("length(v) must be 9^N"))
    size(M,1) == local_dim && size(M,2) == local_dim || throw(ArgumentError("M must be 9×9"))

    x = Vector{Tc}(undef, local_dim)
    y = Vector{Tc}(undef, local_dim)

    @inbounds for leg in 1:N
        stride = ipow(local_dim, N - leg)
        block_size = stride * local_dim
        num_blocks = total_dim ÷ block_size

        for b in 0:num_blocks-1
            base = b * block_size
            for ofs in 0:stride-1
                # gather
                for a in 0:local_dim-1
                    idx = base + ofs + a * stride
                    x[a + 1] = v[idx + 1]
                end
                # y = M * x
                for r in 1:local_dim
                    s = zero(Tc)
                    @simd for c in 1:local_dim
                        s += M[r, c] * x[c]
                    end
                    y[r] = s
                end
                # scatter back
                for a in 0:local_dim-1
                    idx = base + ofs + a * stride
                    v[idx + 1] = y[a + 1]
                end
            end
        end
    end

    return v
end

"""
    phase_space_expectations_fast(ρ::DensityMatrix{<:Complex,3})

Return the length-9^N vector of phase-space expectation values for an N-qutrit density matrix ρ.

This matches the C++ pipeline:
v = vec_rho_tensor_N(ρ)
v ← (M^{⊗N}) v
"""
function phase_space_expectations_fast(ρ::DensityMatrix{T,3}) where {T<:Complex}
    N = ρ.n
    M, _ = build_single_qutrit_M()

    v = vec_rho_tensor_N(ρ.data, N)   # ensure element type matches T
    apply_M_tensor_N_inplace!(v, Matrix{T}(M), N)
    return v
end

"""
    mana(ρ::DensityMatrix{<:Complex,3})

Compute mana of the density matrix `ρ`.

Returns `mana = log2(∑|w| / d)` where `w` are the phase-space expectations
and `d = 3^N` is the Hilbert space dimension.
"""
function Mana(ρ::DensityMatrix{T,3}) where {T<:Complex}
    N = qudits(ρ) # number of qutrits
    dA = 3^N

    w = phase_space_expectations_fast(ρ)

    mana = log2(sum(abs, w) / dA) # mana accumulation

    return mana
end