# Training protocol

## W1/W2 fixed training protocol

The [N300/76-point protocol](W1_W2_PROTOCOL.md) uses a fixed ordered N300 training list, 20 validation trajectories, 76 genuine observation nodes at 0.04-second intervals over t5–8 and a single true t5 initial state. GIFT-Lite uses the first 50 trajectories in the same list. Each method uses fixed hyperparameters and declared seeds, with one training instance per method. W1 includes GIFT, GIFT-Lite, FNO-2D, FNO-3D, U-NO, U-Net and ANIE; W2 includes the six methods other than ANIE.

| Method | Recipe and model selection | Actual supervision coverage |
| --- | --- | --- |
| FNO-2D | seed0, batch20, 500 epochs, Adam lr0.001/wd1e-4, StepLR100×0.5; terminal epoch500 state | Closed-loop curriculum [1, 2, 5, 10, 25, 50, 75], 20 epochs for each of the first six stages, and all 75 steps from epoch121 onward. |
| FNO-3D | seed0, batch10, 500 epochs, the same optimizer/scheduler as FNO-2D; terminal epoch500 state | Jointly outputs 75 future points in one call, with batch-summed spatiotemporal relative L2. |
| U-NO | seed0, batch16, retains the final 12 trajectories, 500 epochs, Adam lr0.001/wd1e-5, StepLR100×0.5; terminal epoch500 state | 20 closed-loop steps from t5, targets through t5.80, batch/time-summed relative L2. |
| U-Net | seed0, batch20, 500 epochs, Adam lr0.001/wd1e-4, StepLR100×0.5; terminal epoch500 state | 4 closed-loop steps from t5, targets through t5.16, batch/time-summed relative L2. |
| ANIE | seed7, effective batch32/micro6, Adam lr1e-4/wd1e-4, delayed CosineT101, cap of 5000 epochs; validation-MSE-best at epoch29, stopping at epoch1029 under patience1000 | Official NS training c=0/y_init path, MSE in original units over all 76 points, including initial-state reconstruction; W1 only. |
| GIFT / GIFT-Lite | generator seed2026072301, branch seed20260820; generator 250+250, branch 400/50/25/25; original stage-specific AdamW and schedules, terminal 500-epoch states | Coarse FD and local RK4; supervision lags of 0.08/0.48 seconds. |

GIFT uses internal RK4 dt = 0.02, saving every 0.04 seconds; W1 advances 150 internal steps. N300 generator/branch update counts are 9500/13600, versus 2000/2600 for GIFT-Lite; identical stage epoch counts do not imply identical computational budgets. The W1/W2 GIFT-Lite source key is `GIFT-N50`; the auxiliary experiment's 50-sample model with the same name belongs to the independent N1000 protocol, with a separate schedule and validation-best selection, and its weights are not interchangeable. Differences in supervision coverage, loss and budget are experimental conditions, so performance differences cannot be attributed solely to architecture. See the [native implementation description](W1_W2_PROTOCOL.md) for network and controller sources.

Confirmation IDs100000–100179 are independent of training and validation, but scores for the two evaluation populations have already been inspected, so this is not a blind evaluation of every method; exploratory IDs1040–1219 are reported separately. Model paths and SHAs are in [CHECKPOINTS.md](CHECKPOINTS.md), and process timings are in [PERFORMANCE.md](PERFORMANCE.md). Configuration sources for the other methods are `prediction_extension/EXTENSION_LOCK.json`.

W3 uses a separate single-trajectory parameter-identification protocol; S1–S3 use the N1000 and GIFT-Lite auxiliary protocols below, with their model and statistical scopes declared separately.

ANIE coordinate and training configuration sources are `prediction_aligned/PROTOCOL_LOCK.json`; source adaptations are detailed in the [ANIE implementation notes](ANIE_ADAPTATION.md).


## Independent auxiliary training protocol (N1000/46 frames/0.02 seconds)

The four supervised prediction baselines use the same 1,000 training trajectories,
46-frame input history and **500 epochs**. Each epoch visits each trajectory once
and selects one reproducibly seeded training window per trajectory. Use the final
epoch, not test performance, to select the released baseline checkpoint.

The four comparable prediction networks use the same 1,000 training trajectories, 46-frame input history and 500 epochs.
Each epoch visits every training trajectory once and selects a time window with a fixed seed.
Each model retains its own prediction procedure and loss computation; equal epoch counts are not claimed to imply equal parameter-update counts or computation.

| Model | Epochs | Physical batch | Gradient accumulation | Expected parameter updates | Training prediction steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| FNO-2D | 500 | 10 | 2 | 25,000 | 150 recursive |
| FNO-3D | 500 | 5 | 2 | 50,000 | 150 jointly predicted |
| U-NO | 500 | 16 | 1 | 31,500 | 20 recursive |
| U-Net | 500 | 20 | 1 | 25,000 | 4 recursive |

U-NO retains its final eight-trajectory batch in every epoch; no trajectory is
dropped. All four schedulers advance once per epoch (StepLR: 100 epochs, factor
0.5). The initial learning rate is 0.001. Weight decay is 0.00001 for U-NO and
0.0001 for the other three. Test trajectories are never used for updates,
normalization, scheduling or checkpoint selection.

## N1000 GIFT and GIFT-Lite auxiliary model training

Two training regimes share the same GIFT architecture. **GIFT** uses all 1,000
prediction-training trajectories; its generator and high-frequency branch each
receive 500 trajectory epochs. This is **500 + 500**, not a 500-epoch total and
not a compute-matched claim. **GIFT-Lite** uses independently trained
50-trajectory models and its own multi-stage schedule; Lite denotes reduced
training data, not a smaller network. W3 identification keeps its separate
protocol below: its generator is trained on, and its coefficients are read from,
the single local training trajectory 0 that the four baselines also use.

GIFT's generator and high-frequency branch are each trained for 500 epochs and reported separately. Each epoch
visits all 1,000 training trajectories once and selects a reproducibly seeded time window from each.
GIFT-Lite uses 50 training trajectories, 20 validation trajectories and validation-based model selection, with no reduction in network size.
Their training budgets also differ, so performance differences cannot be attributed solely to the number of training trajectories.

### Data and model selection

Prediction training IDs are 0–999; GIFT-Lite uses 0–49. Validation IDs are
1000–1039; IDs 1000–1019 contain the complete 0–10 trajectory interval and are
used for checkpoint-validation measurements. Independent test IDs are
1040–1219 (180 trajectories), paired across N64/N96/N128. Identity mapping is
fixed before model evaluation and does not depend on errors or failures.
The identification dataset used by W3 retains its separate local identifiers; all
five W3 configurations identify their parameters from local training trajectory 0.

Full-data GIFT uses **terminal stage states and terminal epoch weights**.
Validation monitors the declared procedure and does not select a model or
alter learning rates. Test observations are not used for parameter updates,
normalization, stage decisions or checkpoint selection.

### Full-data stages and budget

| Component / stage | Trajectory epochs | Physical batch | Parameter updates |
| --- | ---: | ---: | ---: |
| Generator, phase 1 | 250 | 16 (final batch 8) | 15,750 |
| Generator, phase 2 | 250 | 16 (final batch 8) | 15,750 |
| Branch, derivative supervision | 400 | 16 (final batch 8) | 25,200 |
| Branch, short rollout | 50 | 5 | 10,000 |
| Branch, long rollout stage 1 | 25 | 5 | 5,000 |
| Branch, long rollout stage 2 | 25 | 5 | 5,000 |

The generator has 31,500 gradient updates; each branch has 45,200. Its affine
constant/linear fit is a separate analytic operation: one initial fit and one
fit on the same sampled observations at the end of each generator epoch,
501 fits total. These fits are disclosed, not relabelled as gradient updates.
The generator is trained once and reused by the three independently initialized
branches with seeds 20260820, 20260821 and 20260822.

Generator architecture, rank 8, P21 support, four-point centred derivative
numerator on five observed frames, normalized MSE, gradient clipping at 5,
rank-aware affine solver and AdamW are unchanged mathematical components.
The two phase learning rates are 0.01 and 0.003, with within-phase cosine decay
to 2% of the initial value and weight decay 1e-8. The affine fit and the
gradient updates use training observations only.

Branch architecture, Q21 derivative loss, dual-track RK4, local correction,
10-step short rollout, 50-step long rollout with 10-step truncated gradient
segments, and gradient clipping at 1 remain the GIFT mathematical components.
Stage learning rates are 0.0015, 0.0002, 0.00008 and 0.00008; AdamW weight decay
is 1e-6. The derivative stage uses cosine decay to 0.0001. Each new stage
initializes its own optimizer; continuation within a stage restores that
optimizer's complete state. No terminal state is replaced by a better test or
validation checkpoint.

Generator derivative centres are sampled from 2–498. Branch derivative
centres use the same available observation stencil. Rollout anchors are
sampled from 0–450 so every 50-step target is a saved observation within 0–10.
The full-data sampling unit is a trajectory epoch. The reduced-data derivative
stage instead passes over its fixed set of derivative examples.

### Reduced-data GIFT-Lite

GIFT-Lite uses a generator trained for 6,000 + 6,000 random-minibatch
updates on 50 trajectories, with 20 validation trajectories. These updates
are steps, not trajectory epochs. Its three branch runs use 18 derivative
passes, two short-rollout passes and two single-pass long-rollout stages,
5,834 updates per seed. The packaged branches are the validation-selected
weights from their respective fresh-initialization runs.
The shared generator cost is reported once, separately from the three branches.

### Independent commands and continuation

```shell
python -m training.train_gift_generator --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m training.train_gift_predictor --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --low-model ../runs/gift_generator/model.pt --seed 20260820 --output ../runs/gift_branch_20260820
```

Run the branch command separately for each of the three declared seeds. Replace
`--run-training` with `--resume` to restore that run's committed model, optimizer,
scheduler, epoch/stage counters and random state. The default boundary is every
10 epochs and each stage end; `--stop-after-epoch` commits an explicit boundary.
Uncommitted epochs may be repeated after interruption. A published unrelated
generator or a reduced-data checkpoint is not accepted as the full-data training
prerequisite.

All four GIFT training entry points share `--execution auto`: fresh GPU training
uses CUDA graphs, CPU uses eager, and resume retains the journal's backend.
Use `--execution eager` for the explicit reference implementation. This does not
change the full-data 500 + 500 epoch budgets, the reduced-data selection protocol,
or any batch/update counts. Resume checks the model, data, schedule and saved state;
source and environment versions remain descriptive metadata. See
[execution efficiency and measured scope](PERFORMANCE.md).

Full-data timers sum committed epoch time: sampling, forward/backward,
parameter updates, generator affine refits / branch frozen-RHS preparation,
and scheduled validation. They exclude initial input loading/calibration,
checkpoint writing, paused time and discarded uncommitted work. Report the
measured times from each completed run; do not substitute estimated timings.

## Independent runs, timing and continuation

### W3 training protocol for 50 noiseless parameter cases

W3 runs separately on 50 fixed parameter combinations. Each case uses one local ID0 trajectory generated under its true parameters: N64, t=0–10, 501 frames, output interval 0.02 and internal step 0.005. There is no measurement noise or validation trajectory, and trajectories are not reused across cases. Parameters are generated by a locked LHS design: ν∈[0.0075, 0.015] is log-uniform, β and γ∈[0.75, 1.25] are linear, with seed2026100901. The complete CSV SHA and case-specific true values are in the [result package](../results/formal/W3_single_trajectory_20261010/).

GIFT is trained independently for each case with seed2026072301, batch16, 6000+6000 random-minibatch updates, lr0.01/0.003 and P21/rank8. Each stage uses its fixed terminal checkpoint at step 6000; within-training monitoring does not select a model. The final affine fit and parameter readout are performed independently for every case; the budget is 12000 updates per case.

PDE-FIND and PDE-FIND-KC use each case's own ID0 trajectory, the original local-polynomial derivatives, the open90/known4 libraries, STRidge and seed0, without SVD denoising. PINN-SR and PINN-SR-KC respectively use the open90/known4 libraries, seed1234, `paper_ado_v1` and native FP32 with TF32 disabled. Each case has 24,000 training labels, 6,000 held-out labels from the same trajectory and 84,000 physics points. Its budget is 5,000 joint NAdam updates, one native SciPy L-BFGS call, six rounds of 1,000 network NAdam updates and a final 20,000 joint NAdam updates, with terminal-state selection. SciPy limits and measured per-case stopping states are in the result package; reaching a limit does not imply convergence.

The five methods comprise 250 unique method/case slots. No extra seeds, parameter selection or test-error-based weight selection are used; every case is independently initialized, and scientific failures remain attached to their original instances. Complete slot costs, attempts across parallel hosts and interruption boundaries are in the run's FINAL_COST_CLOSURE.

### N1000 prediction baselines

Train each model with its own command, for example:

```shell
python -m scripts.run_training fno2d --run-training --output ../runs/fno2d
python -m scripts.run_training fno3d --run-training --output ../runs/fno3d
python -m scripts.run_training uno --run-training --output ../runs/uno
python -m scripts.run_training unet --run-training --output ../runs/unet
```

The environment, dataset, official external-source paths and user-prepared
`GIFT_EXTERNAL_ADAPTER_ROOT` outside GIFT must be configured first; follow
[EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md).
To continue an interrupted run, add `--resume` and keep the same output directory.
The default committed boundary is every 10 completed epochs; an explicit
`--stop-after-epoch` also commits that boundary. Uncommitted epochs may need to
be repeated. Model, optimizer, scheduler and random state are restored together;
changing source code, data or training settings does not qualify as same-run continuation.

Every committed checkpoint records actual update counts and per-epoch seconds.
The final weights also contain `training_cost`: cumulative committed-epoch wall
time, including batch loading, forward/backward, updates and GPU synchronization.
It excludes setup, checkpoint writing, paused time and discarded unfinished work;
this is not an end-to-end installation or recovery-time measurement. Include this
definition and hardware/runtime details with the experimental cost table. Do not
invent a measured runtime from the expected update counts above.

U-NO retains the upstream bicubic-interpolation backward operation. PyTorch can
report that this CUDA operation is nondeterministic; its warning is not suppressed
and `warn_only=True` does not turn it into a deterministic implementation. Fixed
seeds and complete continuation state preserve the declared procedure, not a
guarantee of bitwise-identical weights across devices or independent CUDA runs.
