# Binary hack generator for q==2
"""
    generate_binary(n) -> (codes, flips)

Generate the length-`n` binary Gray sequence via `i ⊻ (i >> 1)`.
This function is stateless and thread-safe, since each `codes[k]` depends only on `k-1`, so you can
  compute slices or run the loop in parallel without changing the result.

# Returns:
- `codes::Vector{UInt64}` of length `2^n`, where `codes[k] == (k-1) ⊻ ((k-1) >> 1)`.
- `flips::Vector{Int}` of length `2^n-1`, where `flips[k]` is the 1-based
bit position that flipped going from `codes[k]` → `codes[k+1]`.
"""
function generate_binary(n::Integer)
    # total codes
    N = UInt64(1) << n # 2^n

    # build all Gray codes in one go via comprehension
    codes = [i ⊻ (i >> 1) for i = UInt64(0):(N-1)]

    # compute which bit flipped between successive codes:
    #  xor each adjacent pair, then trailing_zeros (0-based) + 1
    diffs = xor.(codes[1:(end-1)], codes[2:end])
    flips = trailing_zeros.(diffs) .+ 1 # one-based index!

    return codes, flips
end

"""
    generate_binary_splitted(n::Integer, rank::Integer, P::Integer)
      → (local_codes::Vector{UInt64},
         local_flips::Vector{Int},
         code_off::Int,
         flip_off::Int)

Partition the global Gray‑code sequence of length 2^n into P contiguous chunks,
and return just the slice for `rank` (0‑based).  You get back both the local
Gray codes and the local “flip” indices, _plus_ the two offsets into the global
arrays.

# Arguments
- `n::Integer`
  Number of bits.  The full sequence has length `2^n`.
- `rank::Integer`
  Which slice you want, in `0:(P-1)`.
- `P::Integer`
  Total number of partitions (MPI size).

# Returns
1. `local_codes::Vector{UInt64}`
   Gray codes for global indices
2. local_flips::Vector{Int}
    Flip indices for global indices
code_off::Int
    0‑based starting index into the full Gray‑code array.
3. flip_off::Int
    0‑based starting index into the full flip‑array.
"""
@fastmath function generate_binary_splitted(n::Int, rank::Int, P::Int)
    @assert 0 ≤ rank < P
    N = UInt64(1) << n   # global length = 2^n

    # ── inline block‐2 partitioning for [0, N) ───────────────────────────────
    q, r     = divrem(Int(N), P)               # q = ⌊N/P⌋, r = N mod P
    code_off = UInt64(rank*q + min(rank, r))   # start idx
    code_cnt = q + (rank < r ? 1 : 0)          # count

    # ── inline block‐2 partitioning for [0, N-1) (the flips) ────────────────
    Nf       = Int(N) - 1
    qf, rf   = divrem(Nf, P)
    flip_off = UInt64(rank*qf + min(rank, rf))
    flip_cnt = qf + (rank < rf ? 1 : 0)

    # ── pre‐allocate exactly what we need ────────────────────────────────────
    local_codes = Vector{UInt64}(undef, code_cnt)
    local_flips = Vector{Int}(undef,    flip_cnt)

    # ── 1) build Gray codes ─────────────────────────────────────────────────
    @inbounds @simd for j in 1:code_cnt
        let i = code_off + (j - 1)
            local_codes[j] = i ⊻ (i >> 1)
        end
    end

    # ── 2) build flip‐positions ─────────────────────────────────────────────
    @inbounds @simd for j in 1:flip_cnt
        let k = flip_off + (j - 1)
            # exactly trailing_zeros( gray(k) ⊻ gray(k+1) ) + 1
            local_flips[j] = trailing_zeros((k ⊻ (k >> 1))
                                        ⊻ ((k+1) ⊻ ((k+1) >> 1))) + 1
        end
    end

    return local_codes, local_flips, Int(code_off), Int(flip_off)
end


"""
    integer_to_gray(val, q, n) -> gray::Vector{Int}

Convert the integer `val` (in `0:q^n-1`) into its q-ary Gray code of length `n`,
using the “additive” algorithm:

1. Compute ordinary base-`q` digits (least significant first) padded to length `n`.
2. Fold into Gray via a running `shift`.

This function is stateless and thread-safe, so its independent of any external state.

# Returns:
- `gray::Vector{Int}` of length `n`, with entries in `0:q-1`.
"""
function integer_to_gray(val::Integer, q::Integer, n::Integer)
    # get ordinary base-q digits, least-significant first, padded to length n
    base_digits = digits(val; base = q, pad = n)
    shift = 0
    gray = Vector{Int}(undef, n)
    @inbounds for i = n:-1:1
        gray[i] = (base_digits[i] + shift) % q
        shift += q - gray[i]
    end
    gray
end

"""
    generate_general(n, q) -> (XTAB, flips)

Generate the reflected-Gray sequence of length `n` over `0:q-1` statelessly, so
its parallelizable: each column `XTAB[:, k]` is computed by calling `integer_to_gray(k-1, q, n)` independently.

# Returns:
- `XTAB::Matrix{Int}` of size `n × q^n`, where column `k` is the `(k-1)`→Gray code.
- `flips::Vector{Int}` of length `q^n-1`, where `flips[k]` is the 1-based digit
    position that changed going from `XTAB[:,k]` → `XTAB[:,k+1]`.
"""
function generate_general(n::Int, q::Int)
    N = q^n

    # build a vector of all Gray codes
    codes = [integer_to_gray(k-1, q, n) for k = 1:N]

    # horizontally concatenate into an n×N matrix
    XTAB = hcat(codes...)            # size (n, N)

    # find, for each transition j→j+1, which digit flipped
    flips = Vector{Int}(undef, N-1)
    @inbounds for j = 1:(N-1)
        # compare columns XTAB[:,j] vs XTAB[:,j+1]
        # and pick the first i where they differ
        flips[j] = findfirst(i -> XTAB[i, j] != XTAB[i, j+1], 1:n)
    end

    return XTAB, flips
end

generate_gray_table(n::Int, ::Val{2}) = generate_binary(n)
generate_gray_table(n::Int, ::Val{Q}) where {Q} = generate_general(n, Q)

"""
    generate_gray_table(n::Int, q::Int)

Wrapper that picks the best method:
- If `q == 2``, calls `generate_binary(n)` to return `(codes::Vector{UInt}, flips)`.
- Otherwise calls `generate_general(n, q)` to return `(XTAB::Matrix{Int}, flips)`.
"""
function generate_gray_table(n::Int, q::Int)
    @assert q ≥ 2 "Base q must be ≥ 2"
    return generate_gray_table(n, Val(q))
end
