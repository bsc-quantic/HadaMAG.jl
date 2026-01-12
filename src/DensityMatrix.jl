using LinearAlgebra

"""# HadaMAG.jl: DensityMatrix
A lightweight container for a mixed quantum state of `n` *q*-dits
(`q` = qudit dimension, 2 for qubits) stored in the computational basis
|0 ... 00⟩, |0 ... 01⟩, ..., |q−1 ... (q−1)⟩.
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
    # total bytes of the buffer
    bytes = Base.summarysize(s.data)
    # user-friendly units
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
        length(s.data),
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
(assuming the usual convention in this library that site 1 is the least-significant digit
in the linear index, consistent with using `q^(s-1)` for site `s`):

- `side = :left`  keeps sites `1:NA` (least-significant):
  `ρA[iA,jA] = ∑_{kB=0}^{dimB-1} ψ[iA + dimA*kB] * conj(ψ[jA + dimA*kB])`.

- `side = :right` keeps sites `(N-NA+1):N` (most-significant).
  `ρA[iA,jA] = ∑_{kB=0}^{dimB-1} ψ[iA*dimB + kB] * conj(ψ[jA*dimB + kB])`.

Returns a `DensityMatrix(ρA; q)` of size `q^NA × q^NA`.
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
        # A = least-significant block (sites 1..NA)
        M = reshape(psi, dimA, dimB) # A × B with entries ψ[iA,kB]
        M * adjoint(M) # (A×B)(B×A) = A×A
    elseif side === :right
        # A = most-significant block (sites (N-NA+1)..N)
        M = reshape(psi, dimB, dimA) # B × A with entries ψ[iA,kB] at M[kB,iA]
        S = adjoint(M) * M # A×A, but corresponds to ρAᵀ
        Matrix(transpose(S)) # transpose to get ρA
    else
        throw(ArgumentError("side must be :left or :right (got $side)"))
    end

    return DensityMatrix(ρA; q)
end
