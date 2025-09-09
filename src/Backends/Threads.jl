module ThreadedBackend # This module provides the low-level kernels for the Threaded backend

using LinearAlgebra
using Random
using HadaMAG

"""
    threaded_chunk_reduce(len, compute_chunk)

Static-scheduled threaded map-reduce over `len` logical chunks.
We pass `(tid, istart, iend)` to `compute_chunk` so it can pick
its *thread-local* scratch buffer by `tid`.

- Partitioning is done once (avoid scheduler migration).
- Each logical `tid` processes a disjoint range [istart, iend].
- Returns the elementwise sum of the tuple results.
"""
function threaded_chunk_reduce(
    len::Integer,
    compute_chunk::Function;
    progress::Bool = false,
    progress_stride::Int = 0,
    hz::Real = 8,
    io::IO = stderr,
)
    nthr = Threads.nthreads()
    tcnt, tdisp = HadaMAG.partition_counts(len, nthr)
    acc = fill((0.0, 0.0), nthr)

    p = progress ? HadaMAG.CounterProgress(len; hz = hz, io = io) : HadaMAG.NoProgress()

    # tiny adaptor so existing 3-arg compute_chunk keeps working
    @inline function call_chunk(f, tid, istart, iend)
        if progress && progress_stride > 0
            return f(tid, istart, iend, p, progress_stride)
        else
            return f(tid, istart, iend, p, 0) # 0 means no progress
        end
    end

    Threads.@threads :static for _ = 1:nthr
        tid = Threads.threadid()
        istart = tdisp[tid] + 1
        iend = tdisp[tid] + tcnt[tid]
        if tcnt[tid] == 0
            acc[tid] = (0.0, 0.0)
        else
            acc[tid] = call_chunk(compute_chunk, tid, istart, iend)
            # coarse progress: count this thread's whole slice as done
            if progress && progress_stride == 0
                HadaMAG.tick!(p, iend - istart + 1)
            end
        end
    end

    HadaMAG.finish!(p)

    a = 0.0;
    b = 0.0
    @inbounds for (a1, b1) in acc
        a += a1;
        b += b1
    end
    return a, b
end

threaded_chunk_reduce(f::Function, len::Integer; kwargs...) =
    threaded_chunk_reduce(len, f; kwargs...)

function SRE(ψ, q; progress::Bool = true)
    L = qubits(ψ)
    dim = 1 << L

    XTAB, Zwhere = generate_gray_table(L, 2)

    # Allocate all the necessary arrays
    TMP1 = zeros(Float64, dim)
    TMP2 = zeros(Float64, dim)
    Xloc1 = zeros(Float64, dim)
    Xloc2 = zeros(Float64, dim)

    # Fill all the arrays by multi-threading
    Threads.@threads for i = 1:dim
        r = real(ψ[i])
        im = imag(ψ[i])
        Xloc1[i] = r
        Xloc2[i] = im
        TMP1[i] = r + im
        TMP2[i] = im - r
    end

    # One `inVR` per thread (thread-local scratch)
    scratch_inVR = [zeros(Float64, dim) for _ = 1:Threads.nthreads()]

    progress_stride = progress ? max(div(length(XTAB), 100), 10) : 0 # update ~100 times

    # Local threaded work — no copies of TMP1/TMP2
    p2SAM, m2SAM = threaded_chunk_reduce(
        length(XTAB);
        progress,
        progress_stride,
    ) do tid, istart, iend, pbar, stride
        HadaMAG.compute_chunk_sre(
            istart,
            iend,
            ψ,
            q,
            Zwhere,
            XTAB,
            TMP1,
            TMP2,
            Xloc1,
            Xloc2,
            scratch_inVR[tid],
            pbar,
            stride,
        )
    end

    return (-log2(m2SAM/dim), abs(1-p2SAM/dim)) # TODO: should we really return 0.0 there?
end

function MC_SRE(
    ψ,
    q,
    Nβ::Int,
    Nsamples::Int;
    seed::Union{Nothing,Int} = nothing,
    progress = true,
)
    Nβ % 2 == 1 || error("Simpson needs an odd number of β points.")
    X = data(ψ)
    dim = length(X)
    L = qubits(ψ)

    # Precompute tmp1/tmp2 ONCE (shared, read-only)
    tmp1 = Vector{Float64}(undef, dim)
    tmp2 = Vector{Float64}(undef, dim)
    HadaMAG.build_tmp!(tmp1, tmp2, X)

    # Per-thread scratch for inVR
    nthr = Threads.nthreads()
    scratch_inVR = [zeros(Float64, dim) for _ = 1:nthr]

    # Outputs per β
    M2 = Vector{Float64}(undef, Nβ)
    M2ADD = Vector{Float64}(undef, Nβ)
    P2 = Vector{Float64}(undef, Nβ)

    base_seed = seed === nothing ? rand(1:(10^9)) : seed

    pbar =
        progress ? HadaMAG.CounterProgress(Nsamples; hz = 8, io = stderr) :
        HadaMAG.NoProgress()
    progress_stride = progress ? max(div(Nsamples, 100), 10) : 0 # update ~100 times

    Threads.@threads :static for i = 1:Nβ
        tid = Threads.threadid()
        β = (i - 1) / (Nβ - 1)
        m2_mean, _m2sq, m2add_mean, p2_mean, _n = HadaMAG.mc_sre_β!(
            X,
            tmp1,
            tmp2,
            scratch_inVR[tid],
            q,
            Nsamples,
            base_seed + i,
            β,
            L,
            pbar,
            progress_stride,
        )
        M2[i] = m2_mean
        M2ADD[i] = m2add_mean
        P2[i] = p2_mean
    end

    x = collect((0:(Nβ-1)) ./ (Nβ-1))
    I_res = HadaMAG.integrate_simpson_uniform(x, M2)
    I_m2ADD = HadaMAG.integrate_simpson_uniform(x, M2ADD)
    m2 = -log2(2.0^(-I_res) + I_m2ADD)
    return m2
end

# TODO: Apply fixes similar to MC_SRE
function mana_SRE2(ψ)
    dim = length(data(ψ))
    L = qudits(ψ)
    Ncores = Threads.nthreads() # Number of threads

    XTAB, Zwhere = generate_gray_table(L, 3)

    # here we know where to act with X_j operator to go from Pauli string number ix, to PS numbered ix+1
    # also we store XTAB which tells us what Pauli string is at position ix
    # constistencY; XTAB[ix] ^= XTAB[ix+1] -- only single 1 at position Zwhere[ix]
    p2SAM = m2SAM = m3SAM = 0.0

    # divide the gray's code (which has 2^N positions) into Ncores, more or less equal patches
    # we store the starting and finishing index corresponding to each of the patches in istart and iend
    istart = [div((i-1)*(length(Zwhere)+1), Ncores) + 1 for i = 1:Ncores]
    iend = [div(i*(length(Zwhere)+1), Ncores) for i = 1:Ncores]

    # The last thread processes until the end of Z (here we mimic Z.size()+1 from C++ by adding one extra element, if needed)
    PS2 = zeros(Float64, Ncores)
    MS2 = zeros(Float64, Ncores)
    MS3 = zeros(ComplexF64, Ncores)

    for j = 1:Ncores
        # Each thread/processes the subrange [istart[j], iend[j]-1]
        ps2, ms2, ms3 = HadaMAG._compute_chunk_mana_SRE(istart[j], iend[j], ψ, Zwhere, XTAB)
        PS2[j] = ps2
        MS2[j] = ms2
        MS3[j] = ms3
    end

    # Now sum the partial sums to obtain the final results.
    p2SAM = sum(PS2)
    m2SAM = sum(MS2)
    m3SAM = sum(MS3)

    println("mana = ", log2(p2SAM)/log2(3.0))

    # SRE is returned here
    return (-log2(m2SAM/dim)/log2(3.0), -1.0*log2(real(m3SAM)/dim)/log2(3.0))
end

end # module ThreadedBackend
