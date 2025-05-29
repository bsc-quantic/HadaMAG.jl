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
    N = UInt64(1) << n

    # build all Gray codes in one go via comprehension
    codes = [i ⊻ (i >> 1) for i = UInt64(0):(N-1)]

    # compute which bit flipped between successive codes:
    #  xor each adjacent pair, then trailing_zeros (0-based) + 1
    diffs = xor.(codes[1:(end-1)], codes[2:end])
    flips = trailing_zeros.(diffs) # zero based index!

    return codes, flips
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
