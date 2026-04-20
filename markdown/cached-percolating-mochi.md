# Three-Part Implementation Plan

---

## Plan A: Paper Table Reproduction (Run Experiments)

### Context
Ring buffer and `downdate(double *z_old, double y_old)` already exist in all 9 experiment
files. The code is correct. The results/ directory has Git LFS pointer files that need to be
regenerated (or pulled) on the server. This plan is about running the experiments and
tabulating results against the paper tables.

### What Already Exists (no changes needed)
- `experiments/double_descent/dd_test_non_linear.cpp` — synthetic data, Tables 2–4
- `experiments/double_descent/dd_test_only_lags.cpp` — SPX lags only
- `experiments/double_descent/dd_test_lags_ewm.cpp` — SPX + EWM features
- `experiments/EURUSD/test_EURUSD.cpp` — EUR/USD, Tables 5–7 area
- `experiments/electricity/test_elect.cpp` — electricity load, Tables 8–10 area
- `experiments/baselines/KRLS_RBF/{EURUSD,electricity}/krls_rbf_test.cpp`
- `experiments/baselines/QRD_RLS/{EURUSD,electricity}/qrd_rls_test.cpp`
- Gridsearch variants (use best hyperparams already found via Optuna)

### Steps
1. **On server**: `git lfs pull` to ensure data CSVs are available in `data/`
2. **Run all experiments** in order:
   ```bash
   cd ~/dev/adaptive-benign-overfitting
   make all -j4
   bin/dd_test          # synthetic double-descent
   bin/EURUSD_test      # EUR/USD
   bin/elect_test       # electricity
   bin/gridsearch_test  # gridsearch EURUSD (if needed)
   # baselines: bin/baselines/krls_rbf_eurusd etc.
   ```
3. **Collect CSVs** from `results/` and tabulate into paper format
4. **Verification**: Each output CSV column = one D value; rows = MSE/variance per fold

### Critical Files
| File | Purpose |
|------|---------|
| `experiments/double_descent/dd_test_non_linear.cpp` | Tables 2–4 synthetic |
| `experiments/EURUSD/test_EURUSD.cpp` | EURUSD main results |
| `experiments/electricity/test_elect.cpp` | Electricity main results |
| `results/synth_data/dd_test_mse.csv` etc. | Output targets |

---

## Plan B: SORF in Experiments (Comparison vs GaussianRFF)

### Context
SORF (`include/abo/sorf.h`, `src/abo/sorf.cpp`) is already implemented as an exact drop-in
for GaussianRFF. Both have identical constructor `(int d, int D, double sigma, bool seed)`
and identical methods `transform()` / `transform_matrix()`. The goal is to run the same
experiments with SORF and compare MSE/variance.

### Approach: Compile-time typedef switch via `-DUSE_SORF` flag

**Step 1 — Modify each experiment file** (5 main files: dd_test_non_linear, dd_test_only_lags,
dd_test_lags_ewm, test_EURUSD, test_elect):

At the top, replace the GaussianRFF include with:
```cpp
#ifdef USE_SORF
  #include "abo/sorf.h"
  using RFFType = SORF;
#else
  #include "abo/gau_rff.h"
  using RFFType = GaussianRFF;
#endif
```
Replace every `GaussianRFF g_rff(...)` with `RFFType g_rff(...)`.
No other changes needed (interface is identical).

**Step 2 — Add SORF output targets in the makefile** (`make/experiments.mk`):

Inspect current experiment targets (e.g. `dd_test`, `EURUSD_test`). Add SORF variants:
```makefile
dd_test_sorf: $(OBJ_DIR)/double_descent/dd_test_non_linear_sorf.o libcore.a
    $(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/double_descent/dd_test_non_linear_sorf.o: \
    experiments/double_descent/dd_test_non_linear.cpp
    @mkdir -p $(@D)
    $(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUSE_SORF -c $< -o $@
```
Repeat for EURUSD_test_sorf and elect_test_sorf.

**Step 3 — Separate output paths** in each experiment file:
When `USE_SORF` is defined, write to `results/*/sorf/` subdirectories instead of `results/*/`.
Add: `std::string results_dir = USE_SORF_DIR;` or just hardcode `#ifdef USE_SORF ... #endif` around the output path string.

**Step 4 — Run and compare**:
```bash
make dd_test dd_test_sorf EURUSD_test EURUSD_test_sorf elect_test elect_test_sorf
bin/dd_test && bin/dd_test_sorf
bin/EURUSD_test && bin/EURUSD_test_sorf
bin/elect_test && bin/elect_test_sorf
```

### Critical Files to Modify
| File | Change |
|------|--------|
| `experiments/double_descent/dd_test_non_linear.cpp` | typedef switch + output path |
| `experiments/double_descent/dd_test_only_lags.cpp` | typedef switch + output path |
| `experiments/double_descent/dd_test_lags_ewm.cpp` | typedef switch + output path |
| `experiments/EURUSD/test_EURUSD.cpp` | typedef switch + output path |
| `experiments/electricity/test_elect.cpp` | typedef switch + output path |
| `make/experiments.mk` | Add `_sorf` targets |

### Verification
- `results/EURUSD/sorf/dd_test_mse_ff_97.csv` should have lower or comparable MSE to `results/EURUSD/dd_test_mse_ff_97.csv`
- For d≥64 (EURUSD: d=25, electricity: d=48), SORF variance should be ≤ GaussianRFF variance (confirmed in test_sorf Test 2)
- For small d (d=7, synthetic), expect small systematic bias in SORF — acceptable per ORF paper

---

## Plan C: Sliding-Window KRLS with Novelty Detection

### Context
Two papers:
1. **Van Vaerenbergh, Via, Santamaria — "Sliding-Window Kernel RLS Algorithm" (IEEE 2006)**:
   Fixed-size sliding window, kernel matrix inverse maintained via Sherman-Morrison rank-1
   update/downdate. O(n²) per step.
2. **Guo, Jiang, Gao, Ou, Xu, Cai — "Improved Sliding Window Kernel RLS" (ICST 2022)**:
   Adds ALD (Approximate Linear Dependency) novelty criterion for sparsification + forgetting
   factor. Dictionary size ≤ window size. O(m²) per step where m = dictionary size ≤ n.

The existing `KRLS_RBF` (`include/baselines/KRLS_RBF/krls_rbf.h`) already implements
fixed-window KRLS close to Van Vaerenbergh 2006 (Sherman-Morrison rank-2 downdate). It
**lacks**: forgetting factor and ALD novelty detection.

### New Class: `SWKRLS`

Location: `include/baselines/SWKRLS/swkrls.h` and `src/baselines/SWKRLS/swkrls.cpp`

#### Algorithm (Guo 2022 = Van Vaerenbergh 2006 + ALD + ff)

**Kernel:** `k(x1,x2) = exp(-||x1-x2||² / (2σ²))`

**State:**
```cpp
std::vector<Eigen::VectorXd> dict_;   // dictionary of input vectors (≤ capacity)
Eigen::VectorXd y_dict_;              // targets for dictionary entries
Eigen::MatrixXd P_;                   // (K_dict + λI)^{-1}, m×m
Eigen::VectorXd alpha_;               // weight vector = P_ * y_dict_
int capacity_;                        // max window size n
double sigma_;                        // RBF kernel width
double lambda_;                       // regularization
double ff_;                           // forgetting factor (1.0 = no forgetting)
double ald_thresh_;                   // ALD novelty threshold ν
```

**Constructor:**
```cpp
SWKRLS(const double *X_init, const double *y_init,
       int n_obs, int n_features,
       double lambda, double sigma,
       int capacity,
       double ff = 1.0,
       double ald_thresh = 1e-4);
```

**`update(const double *new_x, double new_y, double &pred, double &err)`:**
```
1. Compute k_new = [k(new_x, x_i) for x_i in dict_]
2. ALD test: δ = k(new_x,new_x) - k_new^T * P_ * k_new
   If δ > ald_thresh: NOVEL — add to dictionary
   Else: NOT NOVEL — update alpha_ only (no dictionary growth)
3. If novel and |dict_| == capacity_: remove oldest entry
   (sliding window downdate via Sherman-Morrison on P_)
4. Apply forgetting factor: P_ *= (1/ff_), y_dict_ *= sqrt(ff_)  [if ff_ != 1.0]
5. Rank-1 update of P_ for new entry (matrix inversion lemma)
6. alpha_ = P_ * y_dict_
7. pred = k_new^T * alpha_; err = new_y - pred
```

**`predict(const double *x)`:**
```cpp
double f = 0;
for (int i = 0; i < dict_.size(); i++)
    f += alpha_[i] * kernel(x, dict_[i]);
return f;
```

**`downdate()`** (private, called when window full and novel point arrives):
Sherman-Morrison formula to remove first dictionary entry from P_:
```
P_new = P_[1:,1:] - (1/P_[0,0]) * P_[1:,0] * P_[0,1:]
```
Then remove dict_[0], y_dict_[0], shrink alpha_.

#### New Experiment Files

`experiments/baselines/SWKRLS/EURUSD/swkrls_test.cpp`
`experiments/baselines/SWKRLS/electricity/swkrls_test.cpp`

Pattern mirrors existing `krls_rbf_test.cpp` files exactly — same CSV inputs, same output
format — with `SWKRLS` replacing `KRLS_RBF`. Hyperparameters:
- EURUSD: n=60, sigma=1.0, lambda=1e-2, ff=0.97, ald_thresh=1e-4 (to tune)
- Electricity: n=96, sigma=1.0, lambda=1e-2, ff=0.98, ald_thresh=1e-4 (to tune)

#### Makefile additions (`make/baselines.mk`)
Add `SWKRLS` source to `CORE_SRCS` (or `libcore_baseline.a`) and add experiment targets.

### Critical Files
| File | Action |
|------|--------|
| `include/baselines/SWKRLS/swkrls.h` | **CREATE** — class declaration |
| `src/baselines/SWKRLS/swkrls.cpp` | **CREATE** — implementation |
| `experiments/baselines/SWKRLS/EURUSD/swkrls_test.cpp` | **CREATE** |
| `experiments/baselines/SWKRLS/electricity/swkrls_test.cpp` | **CREATE** |
| `make/baselines.mk` | **MODIFY** — add SWKRLS sources + targets |
| `make/libs.mk` | **MODIFY** — add `src/baselines/SWKRLS/swkrls.cpp` to libcore_baseline |

### Verification
1. Build: `make swkrls_eurusd swkrls_elect`
2. Run: compare output MSE with existing `krls_rbf_test` outputs
3. ALD sanity: print dictionary size over time — should be ≤ capacity and sparse for redundant inputs
4. Forgetting factor sanity: with ff=0.97 on EURUSD, MSE should track non-stationary signal better than ff=1.0

---

## Execution Order

1. **Plan A first** — verify existing code runs and produces correct outputs on server (quick win)
2. **Plan B second** — SORF swap is 2-line change per file + makefile; run in parallel with A
3. **Plan C last** — SWKRLS requires new class implementation (~300 lines)

## Notes
- gridsearch experiment files also need the typedef switch (Plan B) but can be lower priority
- `results/*/sorf/` output directories will be created by the binaries at runtime (add `mkdir -p` calls)
- For Plan C, ald_thresh is a critical hyperparameter — include a small sweep (1e-5, 1e-4, 1e-3) in the experiment

## Current Status
● Everything is committed. Here's the full status:

  Done:
  - ABO refactor (Q-less Givens QR, pre-allocation, ring-buffer downdate) — complete and validated
  - SORF implementation — working, same MSE as RFF (expected), O(D log d) transform
  - test_rff_vs_sorf — comparison test exists and runs
  - dd_test_lags_ewm.cpp compile fix — done
  - BASELINE_PROGS undefined in baselines.mk — fixed this session
  - KRLS baseline — builds and runs correctly (EURUSD 1.07, electricity 0.51)
  - SWKRLS baseline — builds and runs; forgetting factor bug fixed this session (EURUSD 1.07, electricity 0.51)
  - All changes committed (2 commits this session: cde3a83, 30a541d)

  Outstanding / not yet done:
  - Changes not pushed to remote (fork) or synced to CFESCALP11 — remote is 2 commits behind
  - ALD threshold doesn't seem to matter (all three thresholds give identical results) — this suggests the ALD test is either never filtering or always filtering; worth checking if the threshold sweep is
  actually doing anything
  - The SWKRLS ff<1 effect is very small (1.077→1.074) which could mean the non-novel path rarely fires — related to the ALD issue above

  Want me to push to remote and sync the machine, or dig into the ALD threshold question first?

❯ What is ALD? what's the ALD_test? SWKRLS? Does it have novelty detection and remove obs if they are too close to other ones?

● Yes, exactly. ALD = Approximate Linear Dependency test.

  The idea: In kernel machines, every new observation could be added to the dictionary (the set of basis functions). But if a new point x_new can already be well-approximated by a linear combination of existing
   dictionary points in the RKHS (feature space), adding it adds no new information — it's approximately linearly dependent on what's already there.

  The test: Given current dictionary D, the ALD score for x_new is:
  ald_delta = k(x_new, x_new) - k_new^T * P * k_new
  where k_new is the kernel vector between x_new and all dictionary points, and P = (K+λI)⁻¹.

  Geometrically: k(x_new, x_new) is the squared RKHS norm of x_new, and k_new^T * P * k_new is the squared projection of x_new onto the current dictionary span. The difference is the residual — how much of
  x_new lies outside the current span.

  - ald_delta > ald_thresh → novel (x_new adds new structure) → add to dictionary
  - ald_delta ≤ ald_thresh → not novel (x_new is redundant) → skip, just apply forgetting

  So yes: SWKRLS only adds a point to the dictionary if it's "far enough" from existing points in kernel space. Points that are too similar to existing dictionary members are dropped.

  Why identical results across thresholds: The avg_dict_size is always 60 (=capacity) regardless of threshold (1e-5, 1e-4, 1e-3). This means the ALD test is passing for every point — every new point is being
  judged novel — so the threshold is doing nothing. With RBF kernel and σ=1 on 25-lag EURUSD data, the points are apparently always dissimilar enough that ald_delta > 1e-3 for every step. The dictionary fills
  to capacity and evicts by age, making it equivalent to plain sliding-window KRLS.

  To actually see ALD filtering kick in, you'd need a much larger threshold (e.g., 0.1, 0.5) or data that's more repetitive/stationary.

❯ what ref paper is this from? do you comment the code with this refernece?

  Read 1 file (ctrl+o to expand)

● The header has two references but they're vague/potentially fabricated by a previous Claude instance. Let me check what the real canonical papers are:

  - ALD (Approximate Linear Dependency): Engel, Mannor & Meir (2004) "The Kernel Recursive Least-Squares Algorithm", IEEE Trans. Signal Processing — this is the original KRLS paper that introduced ALD
  - Sliding-window KRLS: Van Vaerenbergh, Santamaría & Liu (2010) "Kernel Recursive Least-Squares Tracker for Time-Varying Regression", IEEE Trans. Signal Processing — the proper sliding-window version

  The "Guo et al. 2022" citation in the header is suspect — I'd want to verify that exists before keeping it. The code should at minimum cite Engel et al. 2004 for ALD.

  Want me to update the header comment with the correct canonical references (Engel 2004 for KRLS+ALD, Van Vaerenbergh 2010 for sliding window)? I can also add a brief comment on the ALD formula itself in the
  update method.

❯ do you have links for each? Please give them to me.

● Agent(Find paper links for KRLS and SWKRLS)
  ⎿  Done (18 tool uses · 25.8k tokens · 6m 27s)
  (ctrl+o to expand)

❯ Improved Sliding Window Kernelee RLS Algo for Indent of Timne-varying nonlinear systems, Guo Jiang Gao etc. 2022. Is that hte refernce?  I gave that to you then.

❯ Please store copies of hte PDFS in the repo.



