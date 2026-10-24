# Validation scope

Current portable reproduction workflows are in [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md) and [W3_REPRODUCTION.md](W3_REPRODUCTION.md). Original data/model/numerical-file integrity requires byte-for-byte identity; numerical recomputation assesses environment differences through dtype, absolute scales, relative norms and failure masks. Bounded CPU/GPU floating-point differences are not automatically scientific anomalies, and loose thresholds cannot conceal incorrect time axes, wrong models or missing recovery state. Original experiment records remain unchanged.

Evidence for external algorithm boundaries and prediction, weight and result-package equivalence after directory reorganization is in
[EXTERNAL_REPOSITORY_POLICY.md](EXTERNAL_REPOSITORY_POLICY.md). Manual download and adaptation steps are in
[EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md).

Public-file checks also cover textual metadata inside weight containers. Public copies in new model packages must match source files tensor by tensor; when removing source fragments changes serialized bytes, source SHA256 values and public-copy SHA256 values are recorded separately in `ORIGINS.json`, without requiring equality or updating historical experiment model hashes. Original weights, observation data and recorded experimental values must remain unchanged.

## Release validation after source-metadata cleanup

The complete pre-release test run recorded **1090 passed, 12 skipped, 12 subtests passed**, with 45 warnings. It used Python 3.12.14 and PyTorch 2.10.0 + CUDA 12.6; README's Python 3.10.19 describes the original training reference environment and is not the same run. Tests included small CPU continuous-training and pause/resume checks for four actual external prediction networks, without repeating formal 500-epoch training. Skips include explicitly enabled CUDA, TensorFlow and unconfigured external-reference-data tests and are not counted as passes.

An independent read-only review compared all 412 tensors and 49,446,895 scalars in 9 public PT files against original weights: dtype, shape, layout and logical raw bytes matched exactly, without floating-point tolerance. 8 PT files retained original bytes; ANIE removed only 4 preconfirmed source-metadata fields, with all other scientific metadata identical. Archive members, CRC, file sizes and SHA256 passed, and PT serialized strings no longer contained the previous 441-character source text.

A real ANIE inference comparison used the same confirmation trajectory ID100000, RTX A5500 Laptop GPU and complete 76-node timeline, producing shape `[1,76,64,64]`. Original weights and public copies gave bitwise-identical, entirely finite predictions with maximum absolute difference and relative L2 both 0; the original checkpoint hash remained unchanged. This compares one real trajectory over the full timeline without reevaluating all 180 trajectories or retraining.

After final tests, all SHA256 values for 1540 existing weight, result and figure-source files remained unchanged. All sizes and SHA256 values for 1699 files in the independent data package passed verification, alongside roles, shapes, dtypes and 50 W3 cases. This run did not repeat complete-array finite scans or treat file integrity as full scientific recomputation. These checks cover the [current released model package](CURRENT_MODEL_BUNDLE.json) and current working files; they do not establish that all Git history or old release attachments are free of external code.

## W1/W2 validation scope

The [N300/76-point protocol](W1_W2_PROTOCOL.md) checks fixed ordered N300/N50 IDs, 20 validation trajectories, exact subsampling of 76 true nodes, a single true initial state, each method's declared seed/checkpoint identity and terminal-state or validation-best selection. U-NO/U-Net 20/4-step targets, batch/time-sum losses, final partial batches, optimizer/scheduler and random states follow their fixed recipes. W1/W2 GIFT-Lite uses its own N300-protocol generator/branch files (source key `GIFT-N50`), not same-named weights from supplementary N1000 protocols.

Confirmation and exploration populations remain separate, with all 180 trajectories and failure reasons retained per measurement. Scores from both populations have been viewed, so this is not a blind test for all methods. W2 checks elementwise-identical float64 initial-condition parameters across three grids and true 0.04-second nodes; training is N64 only, without fitting target-grid statistics, and ANIE is absent from W2. FNO-3D retains its 75-time-query axis and evaluates the first 25 points; W2 N64 arrays match W1's one-second prefix, with failure decisions also restricted to that prefix.

Model, data and source-file SHA bindings appear in [CHECKPOINTS.md](CHECKPOINTS.md) and result manifests. `scripts.audit_w1_w2_single` recomputes complete predictions for the six existing models and W2, checking 14 W1 method/population measurements and 36 W2 method/grid/population measurements; `scripts.verify_published_results` checks files. Recomputed numerical norms, model identity, byte integrity and actual figure display are distinct validation layers and cannot substitute for each other.

CPU tests include `test_w1_extended_baselines.py`, `test_w2_single_evaluate.py` and single/profile/queue tests. Use `CUDA_VISIBLE_DEVICES=-1` to isolate GPUs and configure external roots for public source. Small shape, one-step or recovery checks validate implementation interfaces rather than complete 500-epoch training or formal prediction performance; synthetic fixtures do not enter formal population metrics or timings.

ANIE's tokenwise space-time correspondence and training configuration are declared in the [ANIE implementation notes](ANIE_ADAPTATION.md); complete training/numerical audits come from `FINAL_TRAINING_AUDIT.json` and `FINAL_RESULT_AUDIT.json` under `prediction_aligned`.

Result evidence appears in the [W1 provenance manifest](../results/formal/W1_flow_prediction_20261007/published.json), [W2 provenance manifest](../results/formal/W2_cross_resolution_20261007/published.json) and `FINAL_RESULT_AUDIT.json`/`FINAL_TRAINING_AUDIT.json` in declared external directories. Independent W3/S1–S3 and N1000 supplementary prediction use their own models and data; interpret the tests below within their declared scopes.

Use the environments and fixed external sources in [SETUP.md](SETUP.md).
External tests also require the user-prepared interfaces outside GIFT selected
by `GIFT_EXTERNAL_ADAPTER_ROOT`; they are not supplied by the repository. Run
commands from the project root. Tests are separate from the full experiments:
a small test passing does not establish that a complete training budget or a
reported scientific conclusion has been reproduced.

The full-data GIFT tests check trajectory-epoch sampling, train/validation/test
identity separation and exact small CPU generator continuation across phase
boundaries. Branch-controller tests cover optimizer, scheduler and RNG recovery
with a bounded surrogate; they are not full native-rollout performance evidence.
Canonical data tests check lossless row copying, metadata ranges, per-trajectory
storage chunks, read-only input checks and correct binding of GIFT versus GIFT-Lite.

## Source and recovery tests

In the prediction/GIFT environment, install the optional test dependencies and
run `python -B -m pytest -q -p no:cacheprovider`. Tests requiring an explicitly
configured external source or another runtime are skipped; skips are not passes.

To exercise the actual four external prediction networks on tiny synthetic CPU
inputs, set `GIFT_EXTERNAL_ROOT` and `GIFT_EXTERNAL_ADAPTER_ROOT`, then use a **new external output directory**:

```powershell
$env:GIFT_BASELINE_TEST_ROOT = '../validation/baseline_cpu'
python -B -m pytest -q -p no:cacheprovider tests/test_baseline_control.py
```

The opt-in cases compare continuous and resumed model, optimizer, scheduler and
random state. They use two tiny epochs, not 500 formal epochs. For another test
execution, choose another output directory; existing evidence is not deleted.

In the separate PINN environment, the NumPy-only sparse-regression tests need
`GIFT_EXTERNAL_ROOT` and the prepared `GIFT_EXTERNAL_ADAPTER_ROOT`, not an earlier project checkout:

```shell
python -B -m unittest discover -s tests -p test_pinn_stridge.py -v
```

These cases check deterministic repetition, unchanged inputs/random state,
the independently recomputed held-out penalized objective, fixed inherited
coefficients, and noiseless unpenalized coefficient recovery. Sparse penalties
can discard true small terms; the noisy tests do not assume perfect recovery.

## Scientific results and packaged files

Use [EXPERIMENT_EXECUTION.md](EXPERIMENT_EXECUTION.md) for full numerical
experiments and [FIGURES.md](FIGURES.md) for their SVGs. Use
[CHECKPOINTS.md](CHECKPOINTS.md) to distinguish fresh training, quick checkpoint
evaluation and same-run continuation. File/hash verification checks integrity;
it is not a substitute for recomputing numerical metrics.

After Git is initialized, inspect the proposed file selection with
`python -B -m scripts.audit_repository`. Once the final files have been staged,
add `--tracked` to limit the audit to tracked files. These are read-only
checks: they do not stage, commit or upload anything.

For an external model attachment, also inspect its contained checkpoints:

```shell
python -B -m scripts.audit_repository --public-file /exports/GIFT-current-models-20261020-source-free.zip
python -B -m scripts.audit_repository --history-risk-ref reference-checkpoints-20260913
```

The first check covers the specified publication file, including nested source
metadata. The second reports known risks in that historical reference without
rewriting it. Earlier `main` history and the old tag still retain executable
adapters; the superseded online model ZIP still retains ANIE source strings and
remains available for provenance. The [new release](../REVIEWER_README.md#reference-weights-and-full-experiments)
provides the cleaned replacement without rewriting that history or physically
removing the old attachment. A passing current-file or replacement-bundle check
must not be presented as certification of all history or all old releases.
Removing source metadata and adding citations do not grant upstream permission;
ANIE's additional authorization remains unconfirmed.

Code tests, checkpoint recovery, file hashes, full training and experiment-metric agreement are verified separately. Synthetic-data tests
do not load existing trained weights or establish that real-data experiments pass. Tests do not automatically train every formal model.
