using Random
using Distributions

# ——— Define the q-ary reflected‐Gray iterator once, for q > 2 ———
"""
    NaryGray{T}(n, q)

An iterator over the `q^n` length-`n` reflected-Gray sequence on `0:q-1`.
On each iteration returns `(code::Vector{T}, changed_index::Int)`, where
`changed_index` is the 1-based position that was incremented or decremented.
"""
struct NaryGray{T}
    n::Int
    q::Int
    code::Vector{T}
    dir::Vector{Int}
end

# Constructor: start at all-zeros, all directions +1
NaryGray(n::Int, q::Int; T = Int) = NaryGray{T}(n, q, zeros(T, n), ones(Int, n))

# First iteration yields the zero-code, with “changed” index = 0
Base.iterate(ng::NaryGray) = ((copy(ng.code), 0), ng)

# Subsequent iterations step one reflected-Gray move
function Base.iterate(ng::NaryGray, _state)
    c, d, q, n = ng.code, ng.dir, ng.q, ng.n
    # find the least-significant digit we can move ±1
    j = findfirst(i -> (0 ≤ c[i] + d[i] < q), 1:n)
    @inbounds c[j] += d[j]
    @inbounds for i = 1:(j-1)
        d[i] = -d[i]
    end
    return ((copy(c), j), ng)
end

# ——— Binary hack generator for q==2 ———
"""
    _gen_gray2(n)

Generate the length-`n` binary Gray sequence using `i ⊻ (i >> 1)`.
Returns `(XTAB, ZT)` where

- `XTAB` is a `Vector{UInt64}` of length `2^n`, each entry the next Gray code.
- `ZT[k]` (length `2^n-1`) is the 1-based bit position that flipped
  going from code `k` to code `k+1`.
"""
@fastmath function _gen_gray2(n::Int)
    N = Int(1) << n          # 2^n
    XTAB = Vector{UInt64}(undef, N)
    ZT = Vector{Int}(undef, N-1)

    prev = UInt64(0)
    XTAB[1] = prev

    for k = 1:(N-1)
        val = UInt64(k) ⊻ (UInt64(k) >> 1)
        diff = val ⊻ prev
        XTAB[k+1] = val
        ZT[k] = trailing_zeros(diff) + 1 # 1-based index !!
        prev = val
    end

    return XTAB, ZT
end


# ——— General q-ary generator for q>2 ———
"""
    _gen_grayN(n, q)

Generate the length-`n` q-ary Gray sequence via reflected-Gray (FKM).
Returns `(XTAB, ZT)` where

- `XTAB` is a `Vector{Vector{Int}}` of length `q^n`, each a digit‐vector.
- `ZT[k]` (length `q^n-1`) is the 1-based digit position that changed
  going from code `k` to code `k+1`.
"""
@fastmath function _gen_grayN(n::Int, q::Int)
    N = q^n
    XTAB = Vector{Vector{Int}}(undef, N)
    ZT = Vector{Int}(undef, N-1)

    ig = NaryGray(n, q)
    i = 1
    for (code, j) in ig
        XTAB[i] = code
        if i > 1
            ZT[i-1] = j # 1-based index of the digit that changed
        end
        i += 1
        i > N && break
    end

    return XTAB, ZT
end

"""
    _gen_grayN_matrix(n, q)

Generate the length-`n` q-ary Gray sequence via reflected-Gray (FKM),
but store everything in a contiguous `n × q^n` matrix instead of a
`Vector{Vector{Int}}`.  This avoids per-iteration small allocations.

Returns `(XTAB, ZT)` where:

- `XTAB[:, k]` is the k-th code (k=1…q^n).
- `ZT[k]` is the 1-based digit-index that changed going from code k→k+1.
"""
@fastmath function _gen_grayN_matrix(n::Int, q::Int)
    N = q^n
    # pre-allocate a matrix of codes and the pivot array
    XTAB = Matrix{Int}(undef, n, N)
    ZT = Vector{Int}(undef, N-1)

    # one single code buffer + direction buffer
    code = zeros(Int, n)
    dir = ones(Int, n)

    # write the all-zero code
    @inbounds XTAB[:, 1] .= code

    for k = 1:(N-1)
        # find the least-significant digit we can move ±1
        @inbounds begin
            for j = 1:n
                # if c[j]+d[j] would stay in 0:(q-1)
                if 0 ≤ code[j] + dir[j] < q
                    code[j] += dir[j]
                    # flip all lower directions
                    for i = 1:(j-1)
                        dir[i] = -dir[i]
                    end
                    ZT[k] = j
                    break
                end
            end
            # dump into the matrix
            XTAB[:, k+1] .= code
        end
    end

    return XTAB, ZT
end


# q == 2
"""
    generate_gray_table(n::Int, ::Val{2})

Specialized for binary: calls `_gen_gray2(n)`.
"""
generate_gray_table(n::Int, ::Val{2}) = _gen_gray2(n)

# q > 2
"""
    generate_gray_table(n::Int, ::Val{Q}) where {Q}

Specialized for Q-ary (Q>2): calls `_gen_grayN(n, Q)`.
"""
generate_gray_table(n::Int, ::Val{Q}) where {Q} = _gen_grayN_matrix(n, Q)


# User‐friendly wrapper ———
"""
    generate_gray_table(n::Int, q::Int)

Generate the full Gray sequence of length-`n` over `0:q-1`, dispatching
on `q` via `Val{q}` so you can call it directly with an integer `q`.

Returns `(XTAB, ZT)` exactly as in the `Val{…}` methods.
"""
function generate_gray_table(n::Int, q::Int)
    @assert q ≥ 2 "Base q must be ≥ 2"
    return generate_gray_table(n, Val(q))
end
