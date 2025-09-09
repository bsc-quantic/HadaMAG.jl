using Base.Threads: Atomic, atomic_add!, atomic_cas!
using Printf: @sprintf

abstract type AbstractProgress end
struct NoProgress <: AbstractProgress end

mutable struct CounterProgress <: AbstractProgress
    total::Int
    done::Atomic{Int}
    last_ns::Atomic{Int}
    throttle_ns::Int
    io::IO
    is_tty::Bool
    finalized::Atomic{Int}  # 0 = not finished, 1 = finished
    step_counts::Int        # print only when >= this many new ticks (0 = disabled)
    last_drawn::Atomic{Int} # count at last emitted line/bar
end

_guess_tty(io::IO) =
    try
        isdefined(Base, :TTY) && (io isa Base.TTY)
    catch
        false
    end

function CounterProgress(
    total::Integer;
    hz::Real = 8,
    io::IO = stderr,
    tty::Union{Nothing,Bool} = nothing,
    pct_step::Real = 0,
    step::Int = 0,
)
    throttle_ns = hz <= 0 ? typemax(Int) : round(Int, 1e9 / hz)
    is_tty = tty === nothing ? _guess_tty(io) : tty
    sc = step > 0 ? step : (pct_step > 0 ? max(1, ceil(Int, total * pct_step / 100)) : 0)
    CounterProgress(
        total,
        Atomic{Int}(0),
        Atomic{Int}(0),
        throttle_ns,
        io,
        is_tty,
        Atomic{Int}(0),
        sc,
        Atomic{Int}(0),
    )
end

@inline tick!(::AbstractProgress, ::Integer) = nothing

# Never render 100% here; leave it to finish! (which is idempotent).
@inline function tick!(p::CounterProgress, n::Integer)
    atomic_add!(p.done, n)
    dnow = p.done[]

    # already finished or reached/passed total -> don't draw
    if p.finalized[] != 0 || dnow >= p.total
        return
    end

    # Step gating: only print when we've advanced by step_counts
    d = dnow
    if p.step_counts > 0
        ld = p.last_drawn[]
        Δ = dnow - ld
        if Δ < p.step_counts
            return
        end
        # snap to the last full step boundary we crossed
        target = ld + (Δ ÷ p.step_counts) * p.step_counts
        target >= p.total && (target = max(p.total - 1, ld))
        # one drawer wins
        if atomic_cas!(p.last_drawn, ld, target) != ld
            return
        end
        d = target
    end

    # Time throttle (set hz=0 to disable)
    t = time_ns()
    lp = p.last_ns[]
    if t - lp <= p.throttle_ns || atomic_cas!(p.last_ns, lp, Int(t)) != lp
        return
    end

    tot = p.total
    pct = 100 * d / max(tot, 1)
    if p.is_tty
        print(
            p.io,
            "\r[",
            repeat("=", Int(clamp(round(Int, pct ÷ 2), 0, 50))),
            repeat(" ", Int(clamp(50 - round(Int, pct ÷ 2), 0, 50))),
            "] ",
            @sprintf("%5.1f%%", pct),
            "  ($d/$tot)",
        )
    else
        println(p.io, @sprintf("%5.1f%%  (%d/%d)", pct, d, tot))
    end
    flush(p.io)
    return
end

finish!(::AbstractProgress) = nothing

function finish!(p::CounterProgress)
    p.done[] = p.total
    if atomic_cas!(p.finalized, 0, 1) == 0
        if p.is_tty
            print(
                p.io,
                "\r[",
                repeat("=", 50),
                "] ",
                @sprintf("%5.1f%%", 100.0),
                "  (",
                p.total,
                "/",
                p.total,
                ")",
            )
            println(p.io)
        else
            println(p.io, @sprintf("%5.1f%%  (%d/%d)", 100.0, p.total, p.total))
        end
        flush(p.io)
    end
    return
end
