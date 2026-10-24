# Checkpoints: train, read or resume

Independent public copies of the current seven inference models and two matching generators can be exported, fully verified and activated on a new machine with `scripts.package_current_models`; see [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md). The [release with source metadata removed](../REVIEWER_README.md#reference-weights-and-full-experiments) provides the [reference model bundle](../REVIEWER_README.md#reference-weights-and-full-experiments); the [archive manifest](CURRENT_MODEL_BUNDLE.json) records its download URL, file size and SHA256. The previous bundle in the [older release](../REVIEWER_README.md#reference-weights-and-full-experiments) still contained four ANIE source-metadata snippets (441 characters) and has been superseded; the old attachment is retained as a provenance record. The public copies remove source metadata and verify exact numerical equality tensor by tensor. Original weights remain read-only, and the original measurement files, current source files and public copies each retain their own file hashes. Current W3 training can start from scratch after creating a new run gate from the formal input package; see [W3_REPRODUCTION.md](W3_REPRODUCTION.md).

Metadata cleanup changes the serialized bytes and SHA256 of the public ANIE copy. It cannot be described as byte-identical to the original file, nor does cleanup grant permission to use, modify or redistribute ANIE weights. See [README](../README.md#references) for provenance and related literature and the [external dependency policy](EXTERNAL_REPOSITORY_POLICY.md) for dependency scope. The table below retains the identities of the current source files; the public copies' new hashes do not overwrite the provenance bindings of historical scientific results.

## W1/W2 model files and provenance

The [N300/76-point protocol](W1_W2_PROTOCOL.md) uses one true initial state, fixed hyperparameters and one model instance per method. Main methods use N300; GIFT-Lite uses the first 50 entries of the fixed ordered list. Model identities, recipes, seeds and data bindings for the other six methods are stored in `prediction_extension/EXTENSION_LOCK.json` and `FORMAL.json` in the same directory. ANIE training configuration and model identity are stored in `prediction_aligned/PROTOCOL_LOCK.json` and `prediction_aligned/training/anie_ns_seed7/ATTEMPT.json`. The following external experiment files are not included in the Git result package.

| Method | Source key and relative path | Current file SHA-256 | Selection rule |
| --- | --- | --- | --- |
| GIFT | `prediction_native/training/GIFT_seed20260820/branch/model.pt` | `fe40059ce9863df79a31e6f3c16e890c577b277519e2fc0a234e4f7014cbf478` | Terminal epoch500 state |
| FNO-2D | `prediction_native/training/fno2d_seed0/terminal_weights.pt` | `6dc9197637184db3f7c412aa82d0630be8eafaa79054cf0401a876fd04ca3b05` | Terminal epoch500 state |
| FNO-3D | `prediction_native/training/fno3d_seed0/terminal_weights.pt` | `286aeea4c54c6810a43ab572b66309c5c3db16fb9c8323286ad581d3968fc0a1` | Terminal epoch500 state |
| ANIE | `prediction_aligned/training/anie_ns_seed7/best.pt` | `3adbe09b412c1a3a214ac06bebde51079298018799b80a685d8cfde4f93abb35` | validation-MSE-best epoch29; W1 only |
| U-NO | `prediction_extension/training/uno_seed0/terminal_weights.pt` | `75334c44d832fab4d5ef525af68c3848ed1392594e1c32885f3c539179919068` | Terminal epoch500 state |
| U-Net | `prediction_extension/training/unet_seed0/terminal_weights.pt` | `ee6918cba6991b904cd6ccaa6738dd992325ad8691ffce5919438e32307eb34c` | Terminal epoch500 state |
| GIFT-Lite | `prediction_extension/training/GIFT-N50_seed20260820/branch/model.pt` | `fe78259d166941e7cab84859669a0a508a0f580f19c7a2a908809d4adeb3ec93` | Terminal epoch500 state |

An external registry resolves the source keys in the table to currently loadable files. Model values, optimizer, RNG, normalizer and observation arrays are bitwise identical to their state at measurement; normalized identity metadata in the current files changes their serialized SHA. The actual weight SHAs used for the original measurements remain in the scientific result manifests, without rewriting the original predictions' provenance bindings. Independent checks are recorded outside the repository in `CANONICAL_ACTIVE_MIRRORS.json` and `CANONICAL_CHECKPOINT_BOUNDARY.json`.

GIFT branches are bound to their matching generators; components cannot be interchanged across configurations:

- GIFT generator: `prediction_native/training/GIFT_seed20260820/generator/model.pt`, SHA-256 `a4489e094c1edd74cabc8ead557a7535ddcdbf45cb67f74aa164af86a3c66c31`.
- GIFT-Lite generator: `prediction_extension/training/GIFT-N50_seed20260820/generator/model.pt`, SHA-256 `4b4f95124604e43c6d880fa10dea48d997f576168321b0f9b2e7313ffed7400c`.

FNO-2D/FNO-3D/U-NO/U-Net all select terminal epoch500 states. ANIE selects epoch29 under the original NS validation-MSE-best rule and trains through epoch1029 under the original early-stopping rule. GIFT and GIFT-Lite use terminal 500-epoch states for both generator and branch, with supervision lags 0.08/0.48 seconds, internal dt0.02 and saving dt0.04. U-NO/U-Net supervise 20/4 steps, and N50 has fewer total updates than N300. ANIE participates only in W1; the six W2 methods freeze their N64-trained weights.

Scores for both evaluation populations have already been inspected, so this is not a blind evaluation of every method; confirmation and exploratory populations are evaluated separately. W1/W2 result packages contain numerical summaries and figure evidence rather than training-continuation files. The existing `artifacts/` weights below belong to independent S1–S3 and N1000 auxiliary protocols; see their individual manifests for metadata and hashes. W3 retains 150 formal terminal models: GIFT and two PINN variants for each of 50 parameter cases. Actual model files, training states and per-case identity bindings reside in external run directories; see the W3 section below. Training recovery requires the model, optimizer, scheduler, RNG and stage state from the same experiment. Final inference weights are not a complete recovery state.

These are separate operations:

| Operation | Input | Training performed? |
| --- | --- | --- |
| Fresh training | Supplied observations and initialized model | Full declared budget |
| Quick evaluation | Packaged trained weights plus test inputs | No |
| Same-run continuation | That run's committed model, optimizer, random and phase state | Remaining uncommitted work |

Downloaded terminal weights are not inserted into an unrelated run's training
journal. See [SETUP.md](SETUP.md) for each model's independent command and the
special PINN L-BFGS phase-boundary recovery rule.

## External implementation location

External-baseline loading requires the user-prepared runtime outside GIFT. Set
`GIFT_EXTERNAL_ADAPTER_ROOT` to its root, such as a sibling
`gift-external-runtime` directory. Official source locations remain as recorded
in the weights and run metadata. Moving the connecting implementation outside
the repository does not change model state keys, architecture, profile or
checkpoint-selection rules. Follow [EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md)
when preparing a new external runtime; matching filenames alone is insufficient
to establish computational equivalence. Existing weight and measurement hashes
remain historical evidence and must not be rewritten for a path migration.

## Training resume

Use the original output directory with `--resume`. Resume restores the model,
optimizer, scheduler, RNG and committed progress, then checks actual model,
data and schedule compatibility. Source hashes, library patch versions, paths,
thread settings and checkpoint intervals do not prevent compatible recovery.
Original checkpoints remain read-only and readable. Publication copies may omit
source-snippet metadata; their separate hashes and exact tensor-equivalence
records distinguish that cleanup from a change to a scientific model.

A complete final checkpoint can also finish an interrupted weights export.
This does not repeat optimization steps. Existing different weights are never
overwritten. To export a baseline checkpoint independently:

```shell
python -m scripts.recover_baseline_export --execute --checkpoint ../runs/uno/checkpoints/epoch_0500_ID.pt --output ../exports/uno.pt
```

ATTEMPT metadata and SHA checks are optional for this file operation. Ordinary
inference uses exported weights and model configuration, without the original
training dataset or optimizer journal.

## Independent N1000 GIFT regimes and auxiliary protocol weights

Full-data prediction GIFT uses `artifacts/gift_full/generator.pt` and
`artifacts/gift_full/gift_seed_SEED.pt`. Its generator and each branch are trained
on 1,000 trajectories for 500 epochs each; their budgets and times are reported
separately. Branch metadata bind the exact generator hash and declared seed.

The full-data generator and all three prediction branches are included with
their completed training histories, budget records and weight hashes. These
are terminal 500-epoch states, not intermediate training checkpoints. Both GIFT
regimes can be evaluated using the commands in EXPERIMENT_EXECUTION.md.

GIFT-Lite uses the preserved clean generator in `artifacts/fixed_k21_n64_unmasked/`
and the three branches in `artifacts/gift/`. It uses 50 training trajectories and
its own multi-stage schedule, not a smaller network. This is not a data-only
ablation with identical training budgets. The current W3 numerical sweep has 50 parameter cases and five methods; it is separate from prediction and GIFT-Lite models. Its fitted models are bound individually to the corresponding case truth and input data.

## W3 identification results for 50 noiseless parameter cases

The current W3 numerical package contains 50 distinct parameter truths, one ID0 trajectory per case and 50 fits for each of five methods. It retains per-case coefficient readouts, full coefficients/support, actual timings and input/source SHAs; no individual case is presented as a single default model for all parameter combinations. See the public [per-case numerical results](../results/formal/W3_single_trajectory_20261010/README.md).

The W3 result package does not contain training checkpoints. Each of the 50 parameter cases has one terminal GIFT model and two terminal PINN models, totaling 150 formal neural models. Actual checkpoints, optimizers, random states and stage records reside outside the repository in `w3_single_trajectory_20261010/training/parameter_*/`, bound through per-case input and model-identity manifests, and are not distributed with the GIFT project repository.

Users must still download external PINN-SR, PDE-FIND and other upstream implementations from the [registered external repositories](EXTERNAL_REPOSITORY_POLICY.md) and adapt them outside the project. This repository retains only instructions, interfaces and references, without these external algorithm implementations.


## Gaussian initial-condition regime (S3)

S3 repeats the same protocol on a separate smooth Gaussian random-field population
and has its own weights under `artifacts/s3_gaussian/`: `gift_full/generator.pt`
and `gift_full/gift_seed_SEED.pt` for full-data GIFT, `gift_lite/generator.pt` and
`gift_lite/gift_seed_SEED.pt` for GIFT-Lite, plus `fno2d/`, `fno3d/`, `unet/` and
`uno/` for the baselines. These are terminal models trained from scratch on the
Gaussian trajectories, not the four-vortex weights. The S3 evaluation rejects any
model whose recorded input population, freshness or frozen-generator binding
differs, so an W1 model cannot silently stand in for an S3 model.

S3 uses the current `gaussian_s3` cohort identity in the separate observation package selected by `GIFT_DATA_ROOT`. Its released model identity metadata is paired to that observation mirror. All learned tensors, optimizer/RNG values and observation arrays retain their original values. `artifacts/CHECKPOINTS.json` lists the actual released file bytes and hashes; the measurement checkpoint hashes in the scientific result provenance still identify the original measured files. The two are separate evidence identities, with equivalence proved outside the repository in `S3_CANONICAL_DATA_MIRROR.json` and `S3_PACKAGED_METADATA_NORMALIZATION.json`.

`artifacts/s3_gaussian/TRAINING_COSTS.json` reports each run's declared budget and
its measured committed interval with an explicit timing scope; those scopes differ
between model families and are not pooled into a single comparable measurement.

## Ordinary and split weight files

`artifacts/CHECKPOINTS.json` records the available weights and SHA-256 hashes.
Most models use ordinary `.pt` or `.npz` files. U-NO uses `weights.json` and
numbered `.part` files in `artifacts/formal/uno/` and in
`artifacts/s3_gaussian/uno/`; keep each of those folders together. U-Net is
95.31 MiB in both regimes and stays a single ordinary file. Only checkpoints that
cannot be distributed as one file within GitHub's per-file limit are split.

The parts contain the **exact bytes of one normal trained PyTorch checkpoint**,
not quantized or altered weights. The loader checks each part and the complete
reconstructed-file hash, then calls `torch.load(weights_only=True)`. Reconstruction
is in memory; it does not create another file or fetch data from the network.

U-NO weights are stored in lossless parts to stay within GitHub's per-file limit for ordinary Git. At runtime they are automatically verified,
concatenated in order and loaded; model parameters, numerical precision and the training algorithm remain unchanged. Do not load an individual part by itself.

An independently trained ordinary `.pt` file is also accepted through W1's
`--uno-model /path/to/model.pt`. To prepare your own completed weights for
distribution, use a new output directory:

```shell
python -m scripts.split_checkpoint --input ../runs/uno/model.pt --output ../weights/uno
```

The default part size is 40 MiB. This storage operation performs no training and
does not establish the checkpoint's scientific accuracy. Reference experiment
commands use their documented budgets; ordinary model loading accepts a compatible
completed custom run and records its actual settings.

## Verification scope

File hash identity, identical tensors, successful same-run recovery and agreement
of experimental metrics are different checks. Runtime/rounding differences can
change generated fields and learned weights. A completed budget is not a claim
of low error or exact sparse-structure recovery. Model-specific timing scopes
are reported alongside the weights rather than pooled as identical measurements.
