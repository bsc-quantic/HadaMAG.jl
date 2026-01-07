using LinearAlgebra

"""# HadaMAG.jl: DensityMatrix
A lightweight container for a mixed quantum state of `n` *q*-dits
(`q` = qudit dimension, 2 for qubits) stored in the computational basis
|0⋯00⟩, |0⋯01⟩, …, |q−1⋯(q−1)⟩.
"""
struct DensityMatrix{T,q}
    data::Matrix{T}
    n::Int
    q::Int
end

"""
    DensityMatrix(mat::AbstractMatrix{<:Complex}; q::Int = 2)

Create a [`DensityMatrix`](@ref) from an existing density matrix `mat`.
Throws `ArgumentError` if `mat` is not square or if its size is not an exact power of `q`.

# Arguments

- `mat`: Square matrix of complex amplitudes.
- `q`: Dimension of each qudit (default = 2 for qubits).

# Returns

- A [`DensityMatrix`](@ref) containing a copy of `mat` and inferred `n` & `q`.
"""
function DensityMatrix(mat::AbstractMatrix{T}; q::Int = 2) where {T}
    size(mat,1) == size(mat,2) || throw(ArgumentError("DensityMatrix must be square"))
    n, ispow = _power_q(size(mat,1), q)
    ispow || throw(ArgumentError("size(mat,1)=$(size(mat,1)) is not a power of q=$q"))
    return DensityMatrix{T,q}(Matrix{T}(mat), n, q)
end

qudits(ρ::DensityMatrix{T,q}) where {T,q} = ρ.n
qudit_dim(ρ::DensityMatrix{T,q}) where {T,q} = ρ.q
data(ρ::DensityMatrix) = ρ.data

Base.size(ρ::DensityMatrix) = size(ρ.data)
Base.getindex(ρ::DensityMatrix, i::Int, j::Int) = ρ.data[i,j]

# Pretty-print summary in the REPL
function Base.show(io::IO, ::MIME"text/plain", s::DensityMatrix)
    # total bytes of the buffer (including any GC-allocated overhead)
    bytes = Base.summarysize(s.data)
    # human-friendly units (optional)
    function _fmt(n)
        for u in ("B", "KiB", "MiB", "GiB")
            if n < 1024
                return "$(round(n, sigdigits=3)) $u"
            end
            n /= 1024
        end
        return "$(round(n, sigdigits=3)) TiB"
    end
    memstr = _fmt(bytes)

    print(
        io,
        "DensityMatrix{",
        eltype(s.data),
        ",",
        s.q,
        "}",
        "(n=",
        s.n,
        ", dim=",
        size(s.data,1),
        ", mem=",
        memstr,
        ")",
    )
end

"""
    reduced_density_matrix(ψ, NA; q=3, side=:right)

Compute the reduced density matrix `ρA = Tr_B(|ψ⟩⟨ψ|)` for a contiguous bipartition
of an `N`-site qudit state (with local dimension `q`), where `dimA = q^NA` and `dimB = q^(N-NA)`.

The keyword argument `side` selects which end of the chain is kept as subsystem `A`
(assuming the usual convention in this codebase that site 1 is the least-significant digit
in the linear index, consistent with using `q^(s-1)` for site `s`):

- `side = :left`  keeps sites `1:NA` (least-significant / fastest-varying block):
  `ρA[iA,jA] = ∑_{kB=0}^{dimB-1} ψ[iA + dimA*kB] * conj(ψ[jA + dimA*kB])`.

- `side = :right` keeps sites `(N-NA+1):N` (most-significant / slowest-varying block),
  matching the C++ convention `i = iA*dimB + kB`:
  `ρA[iA,jA] = ∑_{kB=0}^{dimB-1} ψ[iA*dimB + kB] * conj(ψ[jA*dimB + kB])`.

Returns a `DensityMatrix(ρA; q=q)` of size `q^NA × q^NA`.
"""
function reduced_density_matrix(
    ψ::StateVec{T,q},
    NA::Int;
    side::Symbol = :right,
) where {T<:Complex,q}
    q in (2, 3) || throw(ArgumentError("only q=2 and q=3 are implemented"))

    psi = data(ψ)
    N, ispow = _power_q(length(psi), q)
    ispow || throw(ArgumentError("length(psi)=$(length(psi)) is not a power of q=$q"))
    (0 <= NA <= N) || throw(ArgumentError("need 0 ≤ NA ≤ N (got NA=$NA, N=$N)"))

    dimA = q^NA
    dimB = q^(N - NA)

    ρA = if side === :left
        # A = least-significant block (sites 1..NA under your q^(s-1) indexing)
        M = reshape(psi, dimA, dimB)      # A × B with entries ψ[iA,kB]
        M * adjoint(M)                   # (A×B)(B×A) = A×A
    elseif side === :right
        # A = most-significant block — matches C++ i = iA*dimB + kB
        M = reshape(psi, dimB, dimA)      # B × A with entries ψ[iA,kB] at M[kB,iA]
        S = adjoint(M) * M                # A×A, but corresponds to ρAᵀ
        Matrix(transpose(S))              # IMPORTANT: transpose, NOT adjoint
    else
        throw(ArgumentError("side must be :left or :right (got $side)"))
    end

    return DensityMatrix(ρA; q = q)
end

reduced_density_matrix(ψ::StateVec{Tc,q}, NA::Int; side::Symbol = :right) where {Tc<:Complex,q} =
    reduced_density_matrix(ψ.data, NA; side = side)

# ---------- small integer power ----------
@inline function ipow(a::Int, n::Int)::Int
    n < 0 && throw(ArgumentError("ipow expects n ≥ 0"))
    r = 1
    b = a
    e = n
    while e > 0
        if (e & 1) == 1
            r *= b
        end
        e >>= 1
        b *= b
    end
    return r
end

# ---------- Build single-qutrit M (9×9) ----------
"""
    build_single_qutrit_M()

Return `(M, A_list)` where:
- `A_list[idx]` is the 3×3 phase-point operator A_u (u ≡ idx in 0..8),
- `M` is 9×9 such that for one qutrit: `w = M * vec(rho)` (column-major vec)
    and `w[idx] = tr(A_list[idx] * rho)`.

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
    vec_rho_tensor_N(rho::AbstractMatrix, N::Int)

Map a 3^N×3^N density matrix `rho` (column-major) to a vector `v` of length 9^N
such that each site contributes an index idx9 = 3*i_k + j_k.

This matches the C++ digit convention: the vector legs are ordered MSB→LSB in base-3.
"""
function vec_rho_tensor_N(rho::AbstractMatrix{Tc}, N::Int) where {Tc<:Complex}
    d = 3
    dim = ipow(d, N)
    size(rho, 1) == dim && size(rho, 2) == dim ||
        throw(ArgumentError("rho must be $(dim)×$(dim) for N=$N qutrits"))

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

            # Julia is 1-based; rho is a Matrix so rho[r0+1, c0+1] matches column-major storage
            v[p + 1] = rho[r0 + 1, c0 + 1]
        end
    end

    return v
end

# ---------- Fast in-place application of M^{⊗N} ----------
"""
    apply_M_tensor_N_inplace!(v, M, N)

In-place apply `M^{⊗N}` to `v` (length 9^N), sweeping legs like the C++ code.
`M` must be 9×9.

Leg 1 corresponds to the most significant base-9 digit (same as C++).
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

# ---------- High-level: phase space expectations ----------
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

    v = vec_rho_tensor_N(Matrix{T}(ρ.data), N)   # ensure element type matches T
    apply_M_tensor_N_inplace!(v, Matrix{T}(M), N)
    return v
end

"""
    mana(ψ::StateVec{<:Complex,3}, NA; side=:right)

Compute mana of the NA-qutrit reduced state of the pure qutrit state `ψ`.

Returns `(mana, trρA, tra)` where:
- `trρA ≈ 1` (real)
- `tra = sum(w)/dA` should be ≈ 1 + 0im if conventions match.
"""
function mana(ψ::StateVec{T,3}, NA::Int; side::Symbol = :right) where {T<:Complex}
    ρA = reduced_density_matrix(ψ, NA; side = side)   # DensityMatrix{T,3} of size 3^NA × 3^NA
    dA = 3^NA

    # matches your diag_sum loop
    trρA = real(tr(ρA.data))

    # matches phase_space_expectations_fast(vrho, NA)
    w = phase_space_expectations_fast(ρA)

    # matches mana accumulation + log2(mana/dimA)
    mana_val = log2(sum(abs, w) / dA)

    # matches tra sanity check
    tra = sum(w) / dA

    return mana_val, trρA, tra
end
