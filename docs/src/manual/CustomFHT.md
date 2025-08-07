# Custom FHT Library

Under the hood, `HadaMAG`’s Fast Hadamard Transform (FHT) is powered by the [`FastHadamardStructuredTransforms_jll`](https://github.com/JuliaPackaging/Yggdrasil/tree/master/F/FastHadamardStructuredTransforms) package (a Julia Binary Library artifact built via the [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) infrastructure).  This gives you a portable, pre-built C library (the upstream [FFHT](https://github.com/FALCONN-LIB/FFHT) project) out of the box, with zero fuss on installation.

However, it is important to note that these binaries prioritize **compatibility** over **peak performance**. Therefore, by compiling `FFHT` yourself with optimizations tuned to your CPU (for example passing `-march=native`, enabling link-time optimization, or targeting advanced SIMD extensions), you can often unlock around $10 30 \%$ faster transforms on large vectors. Since `HadaMAG`’s core routines rely heavily on FHT, those gains translate directly into substantial runtime savings.

## Usage and Performance Comparison

Here we show how we can easily switch libraries, and compare the performance of the default JLL-provided library with a custom-compiled one.

```julia
julia> using HadaMAG, BenchmarkTools

julia> L = 22; v = randn(2^L); # around 4 million elements

# Default JLL library
julia> @btime HadaMAG.call_fht!($v, Int32(22))
  8.250 ms (0 allocations: 0 bytes)

# Override with your custom build
julia> HadaMAG.use_fht_lib("/home/user/libffht_julia.so")
[ Info: Using custom FHT library at /home/user/libffht_julia.so

julia> @btime HadaMAG.call_fht!($v, Int32(22))
  6.258 ms (0 allocations: 0 bytes)

# We can also revert back to the default JLL library
julia> HadaMAG.use_default_fht()
[ Info: Reverting to default FHT library

julia> @btime HadaMAG.call_fht!($v, Int32(22))
  8.642 ms (0 allocations: 0 bytes)
```

In this example, we see that the custom-compiled library provides a significant speedup over the default JLL library. You can expect similar results on your own machine, depending on your CPU architecture and the optimizations you apply during compilation.

### API Reference

```@docs
HadaMAG.call_fht!
HadaMAG.use_fht_lib
HadaMAG.use_default_fht
```