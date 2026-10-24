# W1/W2: single initial state, N300, 76 points and fixed recipes

This protocol describes the final W1/W2 data, training methods, evaluation metrics and figure specifications. Machine-readable model/data bindings are in `prediction_extension/EXTENSION_LOCK.json`; training configurations for GIFT, FNO-2D and FNO-3D are in `prediction_native/PROTOCOL_LOCK.json`. ANIE training and coordinate configurations are in `prediction_aligned/PROTOCOL_LOCK.json`. Actual directory names locate sources; see the [model registry](CHECKPOINTS.md) for model files and SHAs and the [experiment record](../EXPERIMENTS.md) for complete numerical results and figures.

## 1. Compared methods

| Method | Training instance and selection | W1 | W2 |
| --- | --- | --- | --- |
| GIFT | generator seed2026072301, branch seed20260820; terminal state | Yes | N64/N96/N128 |
| FNO-2D | seed0; terminal state | Yes | N64/N96/N128 |
| FNO-3D | seed0; terminal state | Yes | N64/N96/N128 |
| ANIE | seed7; validation-MSE-best | Yes | Not included |
| U-NO | seed0; terminal state | Yes | N64/N96/N128 |
| U-Net | seed0; terminal state | Yes | N64/N96/N128 |
| GIFT-Lite | generator seed2026072301, branch seed20260820; terminal state | Reduced-data comparison reported separately | Reduced-data comparison reported separately |

Hyperparameters are fixed in advance, with one training run per method. GIFT's generator and branch are two components of the same method. ANIE selects a checkpoint from the same run using the declared validation MSE and applies the original early-stopping rule; other methods use terminal states.

## 2. Common observations and scoring

- Main training population: `default_rng(2026100304).permutation(1000)[:300]`, with actual ordered IDs saved in the lock. GIFT-Lite uses the first 50 trajectories of this ordered list, rather than original IDs0–49 or samples selected by error.
- Validation: 20 trajectories, original IDs1000–1019; no remaining validation or test trajectory may enter training, statistics or model selection.
- Observations: N64, original vorticity units, t=5, 5.04,…,8, totaling 76 genuine points, extracted exactly from original dense dt0.02 frames 250,252,…,400. There is no temporal interpolation or supervision from omitted dense frames.
- True input: one t=5 field. No method receives 46 historical frames or future truth as input. Training shuffle seed2026100301; GIFT observation-center/anchor seed2026100303; initialization seeds are independent.
- W1: 75 future points. The primary metric averages framewise relative L2 within each trajectory, then averages over the complete 180-trajectory population. Also report t8 relative L2, future75 MSE in original units, curves and failure counts.
- W2: 26 points at 0.04-second intervals over t=5–6; the primary metric covers 25 future points. Also report t6, future25 MSE, frequency-band and failure metrics. Each grid starts from its own single t5 field.
- The known initial state is excluded from primary future metrics; ANIE's original training MSE still includes the model-generated initial-state reconstruction.

Identical availability of 76 observations does not imply identical future lengths in every training loss. Algorithmic objectives, losses, total updates and costs differ; this experiment is neither an architecture-only single-factor ablation nor a comparison under equal GPU budgets.

## 3. Fixed neural-network recipes and actual supervision coverage

| Method | Seed / epoch cap | Physical batch and accumulation | Optimizer / weight decay | Learning rate / scheduler | Selection and training coverage |
| --- | --- | --- | --- | --- | --- |
| FNO-2D | 0 / 500 | 20, no accumulation | Original Adam / 1e-4 | .001; per-epoch StepLR100×.5 | Terminal state; complete closed loop from the single t5 initial state, curriculum [1, 2, 5, 10, 25, 50, 75], first six stages 20 epochs each, 75 future steps from epoch121 |
| FNO-3D | 0 / 500 | 10, no accumulation | Original Adam / 1e-4 | .001; per-epoch StepLR100×.5 | Terminal state; jointly outputs 75 future points in one call |
| ANIE | 7 / 5000 | effective32, micro6; accumulated by actual sample count | Original torch Adam / 1e-4 | 1e-4; CosineT101, eta_min1e-9; advances after completing epoch17 | Same-run validation MSE-best over all 76 points; validation every epoch, original patience1000, ties do not increment counter; full 76-point MSE includes initial state |
| U-NO | 0 / 500 | 16, no accumulation; retains final 12 | Public `UNO/Adam.py` / 1e-5 | .001; per-epoch StepLR100×.5 | Terminal state; 20-step closed loop from t5, supervising t5.04–5.80 |
| U-Net | 0 / 500 | 20, no accumulation | torch Adam / 1e-4 | .001; per-epoch StepLR100×.5 | Terminal state; 4-step closed loop from t5, supervising t5.04–5.16 |

FNO-2D, U-NO and U-Net sum per-field, per-trajectory relative L2 across the batch and then across time. FNO-3D sums relative L2 of each flattened future spatiotemporal field across the batch, without converting it to a sum of framewise errors. No additional `/batch` or time averaging changes gradients supplied to Adam. FNO-3D uses UnitGaussianNormalizer fitted only on the declared inputs/outputs of these 300 training trajectories; other methods retain their declared original-unit paths. ANIE uses physical MSE, without changing it to relative L2. States and spatiotemporal coordinates share space-major/time-minor order, with each token corresponding to the same spatial position and query time during training and evaluation. U-NO's non-reentrant activation checkpointing only recomputes the original forward pass and does not truncate complete closed-loop gradients; U-Net uses no activation checkpointing. There is no teacher forcing or restart from future truth.

Actual N300 update counts are FNO-2D 7500, FNO-3D 15000, U-NO 9500 and U-Net 7500. ANIE has a5000-epoch cap and 10 updates per epoch; it actually stops at epoch1029 after 10290 updates and uses epoch29 validation-MSE-best weights. Every 10 epochs and at the terminal state, U-NO/U-Net run complete 75-step diagnostics on all 20 validation trajectories; model selection and learning-rate schedules retain the prespecified rules. Finite terminal states are retained and evaluated on the complete test set even if validation prediction fails.

FNO and ANIE optimizer and learning-rate recipes come from their respective fixed official NS implementations. U-NO retains the public architecture, Adam and LpLoss, with this project's20-step supervision, per-epoch StepLR and terminal selection; the authors' example uses 40-step supervision, a different scheduler-advancement frequency and validation-best selection. U-Net uses the public architecture with this project's batch size,4-step closed loop, learning rate and selection rule. This experiment compares fixed recipes and is not a complete reproduction of each original paper's experiments.

See [external adaptations](EXTERNAL_ADAPTATIONS.md) for pinned commits and necessary interface changes to public models. This protocol records final training configurations; model files and source identities are in the [model registry](CHECKPOINTS.md).

## 4. GIFT and GIFT-Lite

Both share the cutoff21/rank8 architecture and the same stage/epoch recipe, using terminal states and validation only for monitoring:

| Component / stage | Epochs | Batch | Learning rate / AdamW weight decay | Schedule |
| --- | --- | --- | --- | --- |
| generator 1 /2 | 250 /250 | 16 | .01 /.003; 1e-8 | Independent optimizer per phase; update-wise cosine decay to 2% of initial learning rate |
| branch derivative | 400 | 16 | .0015; 1e-6 | Epoch-wise cosine decay to .0001 |
| branch short | 50 | 5 | .0002; 1e-6 | Constant |
| branch long 1 /2 | 25 /25 | 5 | .00008 /.00008; 1e-6 | Constant; optimizer reset per phase |

Gradient clipping thresholds are 5 for the generator and 1 for the branch. The generator performs affine fitting at initialization and after each epoch,501 fits total. N300 generator/branch update counts are 9500/13600, versus 2000/2600 for N50. GIFT-Lite therefore changes both data quantity and total optimization budget. Here GIFT-Lite denotes the 50-sample configuration of the N300 protocol, with source key `GIFT-N50`. The identically named auxiliary configuration belongs to the separate N1000 protocol and uses 6000+6000 generator updates, a separate branch schedule and validation-based selection; the two sets of weights are not interchangeable.

GIFT FD uses only the 76 genuine observation nodes and dt0.04. Internal RK4 uses dt0.02; nominal 0.1/0.5-second supervision lags select the nearest valid observation node, with ties resolved to the earlier node, giving actual 0.08/0.48-second lags. Short/long training uses 10/50 internal steps. Complete W1 uses 150 internal steps, saving one node every 2 steps for 76 points; W2 uses 50 internal steps and saves 26 points. Numerical safety thresholds are fixed, and integrator rejection counts as failure. Computation uses eager execution.

## 5. Evaluation populations and cross-grid capability

W1/W2 separately report confirmation IDs100000–100179 and exploratory test IDs1040–1219, each with 180 trajectories and no overlap with training or validation. Four methods' confirmation results have already been inspected, so this is not a blind evaluation of every method. Neither population selects training parameters; all trajectories are retained.

The three W2 grids use matching IDs and identical float64 initial-condition parameters for every trajectory. N96/N128 truth is integrated on each respective grid with ETDRK4, internal dt0.005 and padding147/195. Over t5–6, 51 genuine nodes are saved at dt0.02 and then exactly subsampled to 26 points at dt0.04, without temporal interpolation.

GIFT constructs spectral operators on the target grid with frozen weights; FNO-2D/U-NO/U-Net directly accept the target field and recurse 25 times. FNO-3D retains its 75 jointly predicted training-time tokens and takes the first 25 points after complete prediction. Frozen N64 normalizer statistics undergo only deterministic periodic spatial Fourier interpolation, without fitting target-grid data. U-Net's fixed pixel kernels correspond to different physical lengths as the grid changes. ANIE does not participate in W2. See the [W2 implementation](W1_W2_PROTOCOL.md) for interfaces and Nyquist/DFT conventions.

Failures retain the full population: training failures retain their unique slots, and non-finite test predictions or solver rejections retain their actual reasons/prefixes. Undefined full-population metrics are JSON null; finite-only values are diagnostic and are not used to rank successful subsets. A complete finite W2 t5–6 prefix is not counted as a one-second failure merely because its W1 prediction fails after t6. There is no standard deviation across training seeds.

## 6. Timing, figures and numerical provenance

Training-process wall-clock times, timing scopes and provenance for the seven methods are in the [timing record](PERFORMANCE.md). Training time includes data preparation, training, validation, checkpoint I/O and export; GIFT's two components count together as the cost of one method, and W2 does not count training time again. Methods differ in supervision coverage and update budgets, so timing must be interpreted alongside their recipes.

Confirmation and exploratory populations separately retain raw data, metrics, reports and figures. W1 fixes IDs100005/1045 and t5/6/7/8; W2 fixes IDs100110/1150, N128 and t6. PNG previews appear in the text; original SVG curves support editing and export. See the [figure guide](FIGURES.md) for color scales, metric units and failure representations.

Summaries, numerical curves and selected-field arrays are in the [W1 result package](../results/formal/W1_flow_prediction_20261007/published.json) and [W2 result package](../results/formal/W2_cross_resolution_20261007/published.json). Manifests record source numbers, model/data identities and SHA-256; complete weights and prediction HDF files reside in the declared external experiment directories. Independent evaluation commands are in the [execution guide](EXPERIMENT_EXECUTION.md). W3 and S1–S3 have separate protocols and are interpreted individually.
