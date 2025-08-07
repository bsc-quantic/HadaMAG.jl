# Custom FHT Library

Under the hood, `HadaMAG`’s Fast Hadamard Transform (FHT) is powered by the [`FastHadamardStructuredTransforms_jll`](https://github.com/JuliaPackaging/Yggdrasil/tree/master/F/FastHadamardStructuredTransforms) package—a JLL (Julia Binary Library) artifact built via the [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) infrastructure.  This gives you a portable, pre-built C library (the upstream [FFHT](https://github.com/FALCONN-LIB/FFHT) project) out of the box, with zero fuss on installation.

However, it is important to note that those binaries are compiled for **maximum compatibility** rather than **performance**.

By compiling `FFHT` yourself (passing the proper `-march` to exploit your CPU’s full instruction set, enabling Link-Time Optimization (LTO) or Profile-Guided Optimization (PGO), targeting advanced SIMD extensions such as AVX2 or AVX-512, and stripping out debug symbols to permit more aggressive inlining) you can often unlock around $10–30 %$ faster transforms on large vectors. Since our main `HadaMAG` functions rely heavily on FHT operations, those optimizations translate into substantial runtime savings and a game-changing performance uplift.

In the following, we show you how to:

1. Build your own `libffht_julia.so` from the upstream FFHT sources
2. Hook it into `HadaMAG` with `use_fht_lib(...)`
3. Verify the speedup
4. Revert back to the default JLL-provided library if desired

## Usage

1. **Build** your own `libffht_julia.so`, making sure it exports the symbol `:fht_double`.
2. In your Julia session or script:

    ```@repl
    using HadaMAG

    # Point at your custom library (only needs to be called once)
    HadaMAG.use_fht_lib("/home/user/path/to/libffht_julia.so")

    # Now do your transforms
    v = randn(1000000)
    HadaMAG.call_fht!(v, Int32(length(v)))
    ```

3. **Benchmark** to confirm:

    ```@repl
    using BenchmarkTools

    v = randn(1000000)

    # default
    @btime HadaMAG.call_fht!($v, Int32(length($v)))

    # custom
    HadaMAG.use_fht_lib("/home/user/libffht_julia.so")
    @btime HadaMAG.call_fht!($v, Int32(length($v)))
    ```

## Reverting to the default

If you want to revert back to the default JLL-provided library, simply call:

```@repl
HadaMAG.use_default_fht()
```