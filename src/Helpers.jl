# Compute whether `len` is an exact power of `q`, returning exponent and a flag.
function _power_q(len::Integer, q::Integer)
    n = 0
    tmp = len
    while tmp % q == 0
        tmp ÷= q
        n += 1
    end
    return n, tmp == 1
end