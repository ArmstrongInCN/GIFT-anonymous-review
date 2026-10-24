# Independent experiments

For current-protocol execution on a new machine, start with `scripts.reproduce_prediction` in [CURRENT_REPRODUCTION.md](CURRENT_REPRODUCTION.md) (preparation → per-method training → validation → W1/W2 evaluation) and [W3_REPRODUCTION.md](W3_REPRODUCTION.md) (all-50-input gate → training/measurement → 250-slot aggregation). Both workflows directly consume formal data packages without requiring old run directories; local source registries below remain as historical review interfaces.

## W1/W2 execution interfaces and input bindings

The [N300/76-point protocol](W1_W2_PROTOCOL.md) declares fixed seeds and hyperparameters, one training instance per method and a single true t5 field. W1 predicts 75 steps with seven methods; W2 predicts 25 steps on paired N64/N96/N128 grids with six methods, excluding ANIE. Main methods use N300 and GIFT-Lite uses the first 50 entries of the ordered list; U-NO/U-Net receive 20/4 supervised steps, and GIFT uses 0.08/0.48-second supervision with internal dt0.02 and saved dt0.04.

The configuration and model/data-binding directory is `prediction_extension`; its `EXTENSION_LOCK.json` points to `prediction_native/PROTOCOL_LOCK.json` and each method's declared training or weight configuration. Model paths and SHAs are in [CHECKPOINTS.md](CHECKPOINTS.md). Current W1 evaluation loads seven complete models and their declared bindings without automatically retraining.

```powershell
python -m scripts.run_experiment W1 --help
python -m scripts.run_anie_coordinate_refresh --help
```

ANIE configuration, model selection and W1 numerical results are bound to `prediction_aligned`; its space-time token ordering is described in the [ANIE implementation notes](ANIE_ADAPTATION.md). Other methods and W2 use their respective declared directories.

`scripts.run_anie_coordinate_refresh` provides an ANIE training entry point with explicit coordinate alignment. The four-method reference queue in `scripts.run_w1_single` and extended queue in `scripts.run_w1_w2_single` use their respective machine JSON recipes; the former uses ANIE configuration `native_ns_single_v1`, reused by the latter. They cannot replace the `aligned_ns_single_v1` model in the current seven-method main table. Per-model training must read that model's actual machine configuration, rather than treating generic protocol documentation as another queue's recipe.

The current W1 evaluation interface is `experiments.additional.w1_current_evaluate`, loading seven fixed models through an external source registry, with ANIE using its declared space-time-aligned configuration. Inputs explicitly specify truth file, group, confirmation or exploration population, and a new output directory. W2 uses `experiments.additional.w2_single_evaluate`, explicitly binding base/extension locks, models and paired truths. Configuration, data and checkpoint identities within locks must agree. Each entry point's `--help` lists actual arguments. Keep outputs separate from input data and existing measurements; resume with the same configuration directory, retaining committed predictions and progress, and never run two processes simultaneously in one directory. Preserve methods and trajectories with scientific failures; do not replace failed results by changing parameters or resampling seeds.

```shell
python -m scripts.run_experiment W1 --data-file <truth.h5> --group <cohort_group> --cohort confirmation --output ../runs/w1_confirmation --skip-plots
python -m scripts.run_experiment W2 --help
```

`scripts.run_experiment` defaults to the current N300 single-state W1/W2 protocol. The independent N1000, 46-frame-history interface is used only with explicit `--protocol reference`; measurements from that reference protocol cannot populate current W1/W2 main tables.

Confirmation IDs100000–100179 are independent of training and validation, but both evaluation populations' scores have been viewed, so this is not a blind test for all methods; exploration IDs1040–1219 are reported separately. Each population produces complete raw predictions, numeric metrics and figure provenance. Result directories are the [W1 package](../results/formal/W1_flow_prediction_20261007/) and [W2 package](../results/formal/W2_cross_resolution_20261007/). See [implementation notes](W1_W2_PROTOCOL.md) for native networks/training interfaces, [W2 implementation](W1_W2_PROTOCOL.md) for W2 grid/data bindings, and [PERFORMANCE.md](PERFORMANCE.md) for timing scope.

Code and commands directly use W1, W2, W3, S1, S2 and S3, without another identifier space. Current W1/W2 computations use the N300 single-state entry points above; S1–S3 run on their own independent N1000 populations and models, with incompatible weights and input interfaces.

Use the prediction/GIFT environment and external roots from [SETUP.md](SETUP.md).
Each command below runs **one experiment**, reads selected trained weights and
writes to a new output directory. It does not train all models. The optional
`scripts.run_experiment` wrapper configures only its child process: two CPU
threads for prediction/GIFT inference, one for W3 regression/coefficient readout,
one visible GPU, and no global TF32 override. Place `--device cpu` before the
experiment name for an explicitly recorded CPU evaluation.

The following commands are independent. Complete supplied weights allow training to be skipped; when using your own
weights trained from scratch, specify their paths explicitly. Output directories must not overlap data, source code or existing results.

## W3: 50-case noise-free equation-parameter evaluation

The completed W3 campaign evaluates five methods on 50 fixed parameter combinations, each using its own single local ID0 trajectory. It has 250 method/case results and 750 parameter rows. Use the published numerical package to inspect or verify results; the package includes the frozen case list, per-slot outputs, supports, costs and file hashes, but intentionally contains no third-party algorithm source, training coordinator, or replacement single-case default model.

The run-specific dual-host coordinator and exact run/checkpoint state are retained outside the GIFT Git repository. To reproduce training, obtain each external algorithm from the cited upstream repository, install it outside the GIFT directory, apply the written [manual adaptation](EXTERNAL_MANUAL_ADAPTATION.md), and follow the fixed [W3 protocol](W3_PROTOCOL.md). Each training and parameter-readout command must bind the matching case dataset, true coefficients, and checkpoint.

The data/result package is [W3_single_trajectory_20261010](../results/formal/W3_single_trajectory_20261010/README.md). W3 Figure 4 contains three parameter-specific absolute percentage error distributions, each in a single full-range panel: [Figure 4a: ν SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg) ([PNG](../assets/figures/w3_parameter_nu_errors.png)) · [Figure 4b: β SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_beta_errors.svg) ([PNG](../assets/figures/w3_parameter_beta_errors.png)) · [Figure 4c: γ SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_gamma_errors.svg) ([PNG](../assets/figures/w3_parameter_gamma_errors.png)). The [source manifest](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_manifest.json) binds them to the saved parameter rows. Dots have a nominal diameter of 2.1 pt, and the seven lanes are separated by 2.7 pt for a compact point layer. Each real observation is rendered as 12 concentric, stroke-free, same-color transparent discs, using a radial Gaussian alpha profile for a soft edge (sigma 0.42 relative to the radius; nominal central alpha 0.95). This only softens each real marker; it does not apply KDE, smooth the data or add artificial observations. The shared renderer places all exact APE observations on seven fixed horizontal lanes over a Tukey boxplot: solid Q1–Q3 boxes, an extended median line passing through and beyond each box to about 2.1 times its height, and thin whiskers with short caps ending at the outermost actual observations within the inclusive fences [Q1 − 1.5 × IQR, Q3 + 1.5 × IQR], where IQR = Q3 − Q1. Observations outside the whiskers remain included in the same 50 points without extra duplicate flier symbols; means are not marked separately. Solid axes, bold black method names and saturated dark method colors for dots, boxes and whiskers are retained. Median lines use exactly the same color as the box, with a 1.6 pt width and their existing extended length. Horizontal ticks point upward into the plot, while their numerical labels stay below the axis. The bottom spine extends beyond the final tick by 3% of the axis span without changing measured x coordinates or tick positions. Plot text is limited to method names, ticks, the parameter-specific horizontal-axis label and a numerical Median value for each method; see [figure specifications](FIGURES.md).

To reproduce the figure on CPU, choose new output paths:

~~~shell
python -m experiments.formal.w3_equation_identification.plot_parameter_distributions --source results/formal/W3_single_trajectory_20261010/summary/parameter_results.csv --output-dir results/user/W3_parameters/figures --preview-dir results/user/W3_parameters/previews
~~~

This command renders existing numerical results and does not train a model or repeat parameter identification.

## S1 and S2: ablations

```shell
python -m scripts.run_experiment S1 --gift-regime GIFT --output ../runs/S1_GIFT
python -m scripts.run_experiment S1 --gift-regime GIFT-Lite --output ../runs/S1_GIFT_Lite
python -m scripts.combine_regime_results --experiment S1 --gift ../runs/S1_GIFT --gift-lite ../runs/S1_GIFT_Lite --output ../runs/S1
python -m scripts.run_experiment S2 --output ../runs/S2
```

S1 disables the high-frequency branch of each selected GIFT model; S2 compares
recursive local correction on/off. These commands do not start branch retraining. All accept `--low-model`
and three `--gift-model SEED=PATH` arguments for seeds20260820/20260821/20260822.
Use `--lite-low-model` and three `--lite-gift-model SEED=PATH` arguments for explicit
GIFT-Lite overrides. S1 retains independently resumable runs for each training regime;
the combination command verifies both completed reports and joins their tables
without retraining or rerunning inference. S2 evaluates both regimes on one
common independent test population. It preserves failures and
reports finite-only summaries separately; a finite-only mean is not a mean over
the full population. No training trajectories are used as supplementary tests.

## S3: initial-condition distribution

On the experiment machine, select the verified S3 observation mirror before this evaluation:

```powershell
$env:GIFT_DATA_ROOT = '../GIFT-dataset'
```

S3 has a separate [initial-condition distribution protocol](S3_PROTOCOL.md),
Gaussian population and model directory. Use `scripts.run_experiment S3` to
evaluate those weights; its entry point accepts explicit model paths. Four-vortex weights
are not accepted as Gaussian-trained models. The S3 package contains both distributions'
N1000 numerical and fixed-trajectory evidence, independently of W1's N300 measurements.

## Recovery, numerical outputs and SVGs

W1/W2/S1/S2/S3 save checksummed numerical calls at method/seed/grid/cohort
boundaries. Add `--resume` to the identical command with its original output
directory. Completed calls are verified and reused; an interrupted call repeats.
Changed source, data, weights or numerical runtime are rejected. Resumption
prints the new aggregation directory inside the original run; use that directory
as the input to figure rendering. It does not overwrite the prior aggregation.

`summary/metrics.csv` contains summary values; `raw/` preserves arrays or fields.
Keep the completion record and file hashes together. W1/W2 can plot after numeric
completion when `--skip-plots` is omitted. To redraw the five report SVGs separately,
follow [FIGURES.md](FIGURES.md). A figure failure does not invalidate already
committed numerical outputs and does not require retraining.

Check one complete numerical output and, optionally, its separately rendered figures:

The compact results included in the repository are described in [RESULTS.md](RESULTS.md).
They have a separate integrity checker; they are not full experiment directories.

```shell
python -m scripts.verify_results --experiment W1 --result-dir ../runs/W1 --figures ../runs/figures_W1_curve --figures ../runs/figures_W1_keyframes
```

This read-only check verifies completion, file hashes, SVG structure and the
figure-to-report binding. It does not recompute metrics, inspect visual layout or
establish scientific accuracy; those checks remain separate.

Numerical results, tables and SVGs must come from the same completed experiments. Retain nonfinite predictions and failure records;
do not change checkpoints, remove trajectories or adjust color limits to preserve an expected model ranking.
