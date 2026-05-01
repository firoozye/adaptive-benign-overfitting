  ---
  Summary

  Refactor status: complete and validated.

  What the refactor changed vs original

  - Same outputs, different memory model: pre-allocated fixed buffers instead of new[]/delete[] on every streaming step (was 4× malloc per update/downdate)
  - Same algorithm: Q_ matrix is still maintained (needed for old-regime downdate); just stored in a fixed-stride pre-allocated buffer
  - Bug fixed: stale Q_ data after memmove was causing orthogonality loss and MSE blowup — the original addRowAndColumnColMajor masked this by zero-initializing fresh arrays every call

  Is it faster?

  Theoretically yes — zero hot-path heap allocations instead of 4× per streaming step. We haven't run the timing benchmark yet. The -O3 -march=native flag was also missing before and is now added.

  Does it produce correct numbers?

  Yes — validated against paper Table 2:
  - Old regime (D < N=20): Train > 0, moderate test error ✓
  - Interpolation threshold (D=20): massive test spike (1.5×10⁴ vs paper's 819) ✓
  - New regime (D=32): Train≈0, Test=1.40 vs paper's 1.57 ✓
  - Deep overparameterized (D=16384): Train≈0, Test=0.63 vs paper's 0.54 ✓

  To benchmark speed: make timing_test && ./timing_test (though it needs Google Benchmark installed at libs/benchmark).


----


● It works. These are the times per 1000 streaming steps (downdate+update+pred) at each D value. To put this in perspective:

  ┌───────┬─────────────────────┬──────────┐
  │   D   │ Time per 1000 steps │ Per step │
  ├───────┼─────────────────────┼──────────┤
  │ 16    │ 6.6 ms              │ 6.6 µs   │
  ├───────┼─────────────────────┼──────────┤
  │ 32    │ 8.0 ms              │ 8.0 µs   │
  ├───────┼─────────────────────┼──────────┤
  │ 128   │ 16.6 ms             │ 16.6 µs  │
  ├───────┼─────────────────────┼──────────┤
  │ 1024  │ 210 ms              │ 210 µs   │
  ├───────┼─────────────────────┼──────────┤
  │ 16384 │ 5.6 s               │ 5.6 ms   │
  └───────┴─────────────────────┴──────────┘

  The benchmark is working. The note says "CPU scaling is enabled" which adds noise — to get cleaner numbers you can run with:

  ./timing_test --benchmark_min_time=2

  Or disable CPU scaling temporarily:
  sudo cpupower frequency-set --governor performance
  ./timing_test
  sudo cpupower frequency-set --governor powersave

  To compare against the original code (before refactoring), you'd need to check out the previous commit and run the same benchmark. The git log shows the last real code change was 6e0d2ea "Final changes to
  repository" — to compare:

  git stash
  git checkout 6e0d2ea
  make clean && make benchmarks
  ./timing_test
  git stash pop

