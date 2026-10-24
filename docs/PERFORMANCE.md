# GIFT execution efficiency

## W1/W2 recipes and measured costs

GIFT and GIFT-Lite training under the [N300/76-point protocol](W1_W2_PROTOCOL.md) uses eager execution, supervision horizons of 0.08/0.48 seconds, internal RK4 dt = 0.02 and observation-saving interval 0.04 seconds. The independent N1000 controller's auto/cuda-graph policy and acceleration benchmarks are described separately below.

| Method | Single-model training process (seconds) | Minutes |
| --- | ---: | ---: |
| FNO-2D | 5482.334 | 91.37 |
| FNO-3D | 6104.994 | 101.75 |
| GIFT N300, generator and branch combined | 10296.856 | 171.61 |
| ANIE, training for 1029 epochs, MSE-best epoch29 | 21176.742 | 352.95 |
| U-Net N300, 500 epochs | 652.677 | 10.88 |
| U-NO N300, 500 epochs | 8332.408 | 138.87 |
| GIFT-Lite, generator and branch combined | 1911.591 | 31.86 |

The seven training processes total 53957.603 seconds (14.9882 hours). Times include each process's data preparation, training, internal validation, checkpoint writes and export; GIFT's two components are already included in its total, without adding internal timers again. U-NO/U-Net supervise only 20/4 steps, other methods use their own native losses and objectives, and N50 has fewer updates. The table therefore describes costs of fixed recipes rather than architecture-efficiency rankings under a unified compute budget.

| Measurement scope | Process wall-clock (seconds) | Minutes |
| --- | ---: | ---: |
| W1 inference for U-NO/U-Net/GIFT-Lite on two populations, plus scoring and plotting for seven methods | 448.829 | 7.48 |
| Paired N96/N128 confirmation-truth generation | 774.489 | 12.91 |
| Native W2 inference, scoring and figures for six methods on two populations | 2826.245 | 47.10 |

These components time their declared process scopes rather than a unified end-to-end workflow for all W1/W2 training and evaluation. W2 N64 metrics use the one-second prefix of corresponding W1 predictions without counting prediction time again; high-grid FNO-3D outputs all 75 future tokens before evaluating the first 25, so computation is not limited to 25 time tokens. Inference, validation and plotting subtimers within a process are not added again. Environment installation, source acquisition and documentation work are excluded from model-training time.

Cost sources are `source-record:9b0778b4fa631783ad24` (SHA-256 `133fbb3022e0dbe3056d335360f0e78b03df3d990cbfb5a80232c67ec5ca6876`) and `source-record:34daf08966d3f15edc14` (`d3607da57ec27cc6a51270060c5157d92694c5980d561e8ceb93de5477930787`). ANIE training cost comes from `source-record:0458d853302dcbaf7140` (SHA-256 `41aff903c363426402e2ebaa8e64acdc5358d95223c8be7421c55f3e3fa7d4b7`). Other methods' cost scopes are recorded in `FINAL_COST_CLOSURE.json` in their original directories; CPU figure export and similar operations have separate `CPU_EXECUTION_EVENTS.jsonl` records and are excluded from these training timings. Hardware is RTX A5500 Laptop GPU; floating-point and random operators are not guaranteed bitwise-identical across devices.

## Measured W3 costs for 50 noise-free parameter combinations

Each method completed 50 parameter cases. The table reports method-level worker wall-clock totals and per-slot means from FINAL_COST_CLOSURE. Two hosts ran in parallel, so accumulated worker time is neither calendar runtime nor an equal-compute efficiency ranking. Training, regression, parameter readout and auxiliary phases are recorded separately; only fully timed worker wall clocks are accumulated here, while interrupted attempts have separately reported intervals.

| Method | Total worker wall-clock for 50 slots | Per-slot mean |
|---|---:|---:|
| GIFT | 5,313.126 seconds (1.48 hours) | 106.263 seconds |
| PDE-FIND | 12,552.202 seconds (3.49 hours) | 251.044 seconds |
| PDE-FIND-KC | 160.025 seconds (0.04 hours) | 3.200 seconds |
| PINN-SR | 409,616.955 seconds (113.78 hours) | 8,192.339 seconds |
| PINN-SR-KC | 285,430.598 seconds (79.29 hours) | 5,708.612 seconds |


Host worker wall-clock totals are host_A 350,335.529 seconds and host_B 362,737.377 seconds; combined total 713,072.906 seconds (198.08 hours). Parallelism and scheduling mean this differs from elapsed calendar runtime. Auxiliary events such as sampling preparation total 278.298 seconds and are excluded from model-training costs.

The PINN-SR run for parameter041 experienced one infrastructure interruption and then continued from that slot's own checkpoint, without creating another scientific slot or resampling a seed. Evidence bounds the interrupted attempt at 800.370–36924.215 seconds; see [cost closure](../results/formal/W3_single_trajectory_20261010/summary/FINAL_COST_CLOSURE.json) for the interval and complete attempt list. All 250 scientific instances completed; only this interval can be reported for the interrupted attempt, and neither it nor totals including it can be stated as exact values.

Locked budgets and actual timing boundaries appear in the [W3 protocol](W3_PROTOCOL.md), [per-case cost manifests](../results/formal/W3_single_trajectory_20261010/summary/) and run audits. Method objectives, coefficient libraries, optimization procedures and implementation environments differ, so a single wall-clock summary cannot establish a unified efficiency comparison.


## Independent N1000 engine policy

The N1000 GIFT entry points in `scripts.run_training` use the shared `auto` execution policy. Fresh
GPU training uses `cuda-graph` to replay the same tensor operations with less
CPU dispatch overhead; CPU uses `eager`. `--execution eager` selects the explicit
reference training path. Prediction remains eager by default. No extra package
is needed. CPU training uses the same model and declared budget with eager execution.

The N1000 `scripts.run_training` GIFT entry points share an `auto` execution policy: new GPU training defaults to `cuda-graph`,
while CPU uses `eager`; `--execution eager` explicitly selects the reference training path. Prediction defaults remain unchanged.
The optimization preserves models, training-data volumes, epoch counts, batch sizes,
AdamW, learning-rate schedules, losses, RK4 steps and recursion truncation positions, without enabling mixed precision.

## Use

Full-data GIFT and reduced-data GIFT-Lite use the same training engine. Each
model still has its own command, scientific schedule and complete-state journal:

```shell
python -m scripts.run_training gift_generator --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m scripts.run_training gift_predictor --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --low-model ../runs/gift_generator/model.pt --seed 20260820 --output ../runs/gift_branch_20260820
python -m scripts.run_training gift_low --run-training --condition noise_000 --output ../runs/gift_lite_generator
python -m scripts.run_training gift_branch --run-training --low-model ../runs/gift_lite_generator/gift_main.pt --seed 20260820 --output ../runs/gift_lite_branch_20260820
```

Continue with `--resume` and the same output directory. The model, optimizer,
scheduler, RNG and progress are restored from the latest committed boundary.
Graphs are rebuilt; no CUDA graph or device address is serialized. Source and
environment records describe the run rather than prevent code maintenance.
Actual model, data and schedule compatibility is checked before continuing.

Checkpoint recovery uses the original output directory. For `gift_low`,
`gift_generator` and `gift_predictor`, replace `--run-training` with `--resume`;
for `gift_branch`, retain `--run-training` and add `--resume`.
Execution graphs are rebuilt after recovery without saving device memory addresses. Source and environment changes do not prevent recovery of compatible states;
actual runtime settings are recorded. Initial graph construction counts toward training-epoch time.

The reduced-data branch command retains `--run-training` when adding `--resume`.
The clean/noisy reduced-data generator keeps its best-validation phase selection,
random-minibatch sampling and affine refit schedule. GIFT-Lite branches keep their
derivative/short/long selection and optimizer/loader resets. These protocols are
not replaced by the full-data trajectory-epoch schedule. W3 evaluation, published
weights and scientific settings are not changed by execution selection.

Use `--continue-from PARENT_RUN` and a new output directory when intentionally
starting a separate continuation. It copies the committed model, optimizer,
scheduler, RNG and history without a training update. No transition certificate
is required. The parent remains unchanged. Uncommitted work restarts from the
last boundary; paused time is not counted as training time.

W1/W2/S1/S2/S3 accept `--gift-execution cuda-graph`. This only changes GIFT
execution, including the GIFT-Lite inference arm. External algorithms and W3
are unaffected. Inference retains per-trajectory failures, correction gates and
RHS evaluation counts; detailed leakage diagnostics are optional. Its reported elapsed
time includes graph construction. Very short predictions can be slower because
the initial graph cost is not yet amortized.
Reported RHS counts describe the simulated RK4 steps. Graph warmups are extra
discarded evaluations, not additional simulated steps or optimizer updates;
their time is included in graph setup and total elapsed time.

## Measurements

Hardware: RTX A5500 Laptop GPU (16 GiB), Intel i9-12950HX, Windows,
PyTorch 2.10.0 + CUDA 12.6; 16 Torch CPU threads. Original float32/complex64
arithmetic and TF32 settings are retained. All GIFT training entry points disable
cuDNN benchmarking, select deterministic cuDNN algorithms and fix the cuBLAS
workspace before tensor work. Seeding alone is insufficient for reproducible
convolution backward operations. Full-data/generator determinism policies are
retained; the reduced-data branch uses this same explicit runtime policy.
Journals bind these settings, and numerical agreement across different hardware
or library versions is not guaranteed to be bitwise. Affine-fit statistics
remain float64/complex128 with the original chunk size and rank threshold.

The following measurements visit 60 observed training trajectories per repeat,
with three repeats and alternating backend order. Each timing includes host-to-
device copies, forward/backward, clipping, AdamW and scalar loss reporting.
It excludes initial graph setup, data preparation, validation and affine refits.
These are component benchmarks, **not completed 500-epoch training times**.

| Stage | Eager median (seconds) | Graph median (seconds) | Speedup |
| --- | ---: | ---: | ---: |
| Generator gradient training | 0.0759 | 0.0273 | 2.78× |
| Branch derivative training | 0.0575 | 0.0338 | 1.70× |
| Branch short recursion | 6.1704 | 1.3059 | 4.73× |
| Branch long recursion | 31.5045 | 6.4316 | 4.90× |

Graph setup, including two warmups, took approximately 0.23, 0.32, 1.42 and
6.30 seconds respectively for the measured batch shapes. Results depend on
hardware, thermal state, batch shape and runtime; they are not universal ratios.
Final model parameters and aggregate losses matched bitwise in all repeats.

An additional seven-trajectory, 50-step inference check retained complete
diagnostics. With correction enabled, total elapsed-time speedups including
graph setup were 2.35× at N64, 1.92× at N96 and 1.50× at N128. N64 used observed
training fields; the larger grids used explicitly resized timing/safety fixtures,
not cross-resolution accuracy data. Predictions, failure locations/reasons,
trigger counts, leakage measurements and all scientific report fields matched
the eager reference. Artificial failures were also checked: a failure-heavy N128
case was about 6% slower, illustrating why this backend remains optional.

Additional numerical stress tests covered 1,000 real trajectory inputs per
branch stage (63 derivative updates; 200 short/long updates), comparing
raw gradients, clipped gradients, model parameters and complete optimizer state
after each update. The paired stress harness also checked identical controlled
scheduler states; it is not a newly completed formal epoch. Two further branch
initialization seeds were checked on 21 trajectories each, including single-row
tail batches. All comparisons were bitwise equal. Fresh-process continuation was checked separately through the
actual training entry points on explicitly marked, short-budget fixtures,
including phase changes and short final batches. This does not replace a full
500 + 500 epoch accuracy run.

Independent short-budget continuation and complete-budget equality checks are engineering evidence for the execution backend. They do not replace the current W3 50-case noise-free measurements or contribute additional statistical training seeds. Detailed source versions and numerical check records are retained outside the repository.

All three reduced-data branches were also trained from scratch for the complete
5,834-update schedule. Their committed training-plus-validation times were
205.49, 233.10 and 221.20 seconds for seeds 20260820, 20260821 and 20260822.
These figures include graph setup and omit shared generator pretraining, initial
data preparation and checkpoint/export I/O. An independent second full-budget
run of seed 20260820 matched the first run's selected parameters, optimizer,
scheduler, RNG and numerical history bitwise, excluding elapsed-time fields.
The repeat is a reproducibility check, not a fourth statistical seed.

## What is optimized

- Cache only constant frequency grids, masks and table indices. Learned
  generator tables and branch coefficient fields are recomputed after updates.
- Capture forward/backward tensor work separately for each phase/batch shape.
  Gradient clipping, AdamW and scheduler steps remain ordinary PyTorch calls.
  A short final batch has its own graph: no padding or changed sample weighting.
- For full training batches (16 derivative trajectories or 5 rollout trajectories),
  pack differentiable spectral-weight layouts once per backward segment, build
  disjoint Fourier output bands by concatenation, and batch the four coefficient
  projections. Parameters keep their ordinary storage and checkpoint format.
  Packed views are rebuilt every segment and never reused across updates. Other
  batch shapes use the reference graph because layout changes can affect the
  last bits of gradients. Default prediction and validation do not use these
  training-only kernels.
- Use the same local-correction tensor core in training and audited inference.
  Training omits diagnostic reports it does not consume and accumulates safety
  flags on the device. A failed batch is rejected **before** the optimizer step.
- Capture the complete projected inference RK4 step, including its eight
  leakage measurements. Correction, failure isolation and output sampling retain
  their original behavior. Retain at most two active batch shapes; additional
  failure-reduced shapes use eager execution without padding trajectories.

Data verification/loading was about 35 seconds once at startup on the measured
system. Per-epoch sampling was about 0.05–0.07 seconds; derivative target/frozen
RHS preparation about 0.32 seconds; a 1,000-state affine refit about 0.06 seconds.
These are not the dominant long-recursion bottleneck. Hash verification, observed
derivative construction, affine-fit precision and checkpoint durability are not
weakened for speed. Increasing batch size, reducing RK4 stages, enabling lower
precision or rewriting external algorithms is deliberately outside this backend.

Training-specific layouts do not reduce RK4 steps, branch calls or backward segments. A long-recursion batch still contains 50 RK4
time steps, 200 branch forward calls and 5 backward segments. Final partial batches are neither padded nor dropped and use the original graph execution path.
Layout copies connect to original parameters through ordinary autodifferentiation, without detaching weights or implementing custom backward operators.

## Reproduce timing

Use the same configured environment and explicitly selected model files:

```shell
python -m scripts.benchmark_gift --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --generator ../runs/gift_generator/model.pt --branch ../runs/gift_branch_20260820/model.pt --rows 60 --repeats 3 --output ../benchmarks/gift.json
```

This writes a new JSON receipt and performs disposable optimizer updates only.
It does not overwrite weights, resume formal training, or update experiment
tables. An explicitly supplied same-run training checkpoint may also be used
for disposable timing. Benchmark speed never substitutes for accuracy evidence.

For graph buffer/RNG tests, set `GIFT_RUN_CUDA_TESTS=1` and run
`python -m pytest tests/test_gift_acceleration.py -q`. Ordinary tests exercise
CPU fallback and reference behavior without requiring a GPU.

See the [PyTorch CUDA graph documentation](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs)
for fixed-address/shape requirements and the
[profiler documentation](https://docs.pytorch.org/docs/2.10/profiler.html) for
operator profiling. Numerical equality measured on one configuration does not
guarantee bitwise equality across every GPU, library version or future run.
