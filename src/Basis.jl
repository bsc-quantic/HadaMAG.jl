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

Partition the length-`2^n` binary Gray-code sequence into `P` contiguous chunks
(of global indices) and return the slice for this `rank` (0-based).

For this slice:

  • `local_codes[j]` is the Gray code for the global index `code_off + (j-1)`.
  • `local_flips[j]` is the bit position that flips between
    `local_codes[j]` and `local_codes[j+1]`, for `j = 1:(length(local_codes)-1)`.

Cross-chunk transitions (between the last code of slice `rank` and the first
code of slice `rank+1`) are *not* included in `local_flips`. This makes the
result suitable for incremental Gray walks that restart from a known mask at
the beginning of each chunk (e.g. per-rank masked FHT sweeps).

# Arguments

- `n::Integer`
  Number of bits. The full Gray sequence has length `2^n`.

- `rank::Integer`
  Which slice you want, in `0:(P-1)`.

- `P::Integer`
  Total number of partitions (e.g. MPI size).

# Returns

1. `local_codes::Vector{UInt64}`
   Gray codes for global indices in this slice.

2. `local_flips::Vector{Int}`
   For each *internal* transition within the slice, the 1-based bit index
   that flips between successive Gray codes.

3. `code_off::Int`
   0-based starting index into the full Gray-code sequence.

4. `flip_off::Int`
   0-based starting index aligned with `code_off` (for incremental use).
"""
@fastmath function generate_binary_splitted(n::Int, rank::Int, P::Int)
    @assert 0 ≤ rank < P
    N = Int(1) << n              # total number of Gray codes

    # Partition codes [0, N) across ranks
    code_counts, code_displs = HadaMAG.partition_counts(N, P)
    code_off = code_displs[rank+1]      # 0-based global index
    code_cnt = code_counts[rank+1]

    # 1) local Gray codes
    local_codes = Vector{UInt64}(undef, code_cnt)
    @inbounds @simd for j = 1:code_cnt
        i = UInt64(code_off + (j - 1))  # global integer index
        local_codes[j] = i ⊻ (i >> 1)
    end

    # 2) local flips: only internal transitions within this rank's slice
    # transitions are between global indices [code_off .. code_off+code_cnt-2]
    flip_cnt = max(code_cnt - 1, 0)
    local_flips = Vector{Int}(undef, flip_cnt)
    @inbounds @simd for j = 1:flip_cnt
        g0 = local_codes[j]
        g1 = local_codes[j+1]
        local_flips[j] = trailing_zeros(g0 ⊻ g1) + 1
    end

    # we can return code_off as the "offset" for both; flips are aligned to codes
    flip_off = code_off

    return local_codes, local_flips, code_off, flip_off
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

"""
    generate_general_splitted(n::Int, q::Int, rank::Int, P::Int)
      → (local_XTAB::Matrix{Int},
         local_flips::Vector{Int},
         code_off::Int,
         flip_off::Int)

Partition the q-ary Gray-code sequence of length `q^n` into `P` contiguous
chunks of global indices and return the slice for this `rank` (0-based).

For this slice:

  • `local_XTAB[:, j]` is the Gray code for global index `code_off + (j-1)`.
  • `local_flips[j]` is the digit position that changes between
    `local_XTAB[:, j]` and `local_XTAB[:, j+1]`, for
    `j = 1:(size(local_XTAB,2)-1)`.

Cross-chunk transitions (between the last column of slice `rank` and the first
column of slice `rank+1`) are *not* included in `local_flips`. This matches
the binary `generate_binary_splitted` behavior and is suitable for per-rank
q-ary Gray walks that restart from a known state at the beginning of each chunk.

# Arguments
- `n::Int`: number of digits; full sequence has length `q^n`.
- `q::Int`: base (≥ 2).
- `rank::Int`: which slice you want, in `0:(P-1)`.
- `P::Int`: total number of partitions.

# Returns
1. `local_XTAB::Matrix{Int}` (size `n × code_cnt`):
     Gray codes for global indices in this slice.
2. `local_flips::Vector{Int}`:
     For each *internal* transition within the slice, the 1-based digit index
     that changes between successive Gray codes.
3. `code_off::Int`:
     0-based starting index into the full Gray-code sequence.
4. `flip_off::Int`:
     0-based starting index aligned with `code_off` (by construction
     `flip_off == code_off`).
"""
@fastmath function generate_general_splitted(n::Int, q::Int, rank::Int, P::Int)
    @assert 0 ≤ rank < P
    @assert q ≥ 2

    # total number of codes (assumes q^n fits in Int)
    N = q^n

    # ── partition codes [0, N) exactly like in the binary version ───────────
    code_counts, code_displs = HadaMAG.partition_counts(N, P)
    code_off = code_displs[rank+1]   # 0-based
    code_cnt = code_counts[rank+1]

    # ── preallocate local matrix and flips ───────────────────────────────────
    local_XTAB = Matrix{Int}(undef, n, code_cnt)

    # 1) build local columns (Gray codes for this chunk)
    @inbounds for j = 1:code_cnt
        # global 1-based column index
        k = code_off + j
        code_vec = integer_to_gray(k - 1, q, n)  # length-n vector
        local_XTAB[:, j] = code_vec
    end

    # 2) local flips: only internal transitions within this chunk
    flip_cnt = max(code_cnt - 1, 0)
    local_flips = Vector{Int}(undef, flip_cnt)

    @inbounds for j = 1:flip_cnt
        # first position i where column j and j+1 differ
        pos = 0
        for i = 1:n
            if local_XTAB[i, j] != local_XTAB[i, j+1]
                pos = i
                break
            end
        end
        @assert pos != 0  # valid reflected Gray code must differ in some position
        local_flips[j] = pos
    end

    # flips are aligned to the same global offset as codes
    flip_off = code_off

    return local_XTAB, local_flips, code_off, flip_off
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
