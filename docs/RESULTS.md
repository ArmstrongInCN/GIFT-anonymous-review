# Reading results

## W1/W2 methods, populations and metrics

W1 compares GIFT, GIFT-Lite, FNO-2D, FNO-3D, U-NO, U-Net and ANIE; W2 compares the six methods excluding ANIE. Main methods use 300 N64 training trajectories, 20 validation trajectories, 76 true nodes every 0.04 seconds over t5–8, and each trajectory's own true t5 initial state. GIFT-Lite uses the first 50 entries of the ordered N300 list. Each method trains one model instance with fixed hyperparameters and its declared seed. Recipes, selection rules and weight bindings are in the [training protocol](W1_W2_PROTOCOL.md); complete result tables and timings are in [EXPERIMENTS.md](../EXPERIMENTS.md).

The primary W1 metric is the mean framewise relative L2 over 75 future points; W2 evaluates 25 future points over t5–6 on N64/N96/N128. First average future nodes within each trajectory, then average all 180 trajectories in the population; endpoint error, MSE in original vorticity units and failure counts are also reported. Each method has one training instance, so no means or standard deviations across training seeds are reported.

The two evaluation populations are reported separately, without pooling:

- Confirmation IDs100000–100179: 180 trajectories, disjoint from training and validation. Results from both evaluation populations have been viewed and cannot be regarded as a blind test for all methods.
- Exploration IDs1040–1219: 180 trajectories, used only for exploratory evaluation.

W2 uses matching IDs and elementwise-identical float64 initial-condition parameters. N96/N128 truths are genuinely integrated on their own grids and then exactly subsampled at 0.04-second nodes. Models train only on N64; inference neither fits target-grid weights nor reads test normalization statistics.

## Method differences and results

Shared inputs and evaluation populations do not imply equal losses or budgets. U-NO supervises only 20 steps to t5.80 and U-Net only 4 steps to t5.16; FNO, ANIE and GIFT use their declared full-window, curriculum or local objectives. GIFT-Lite retains GIFT's phase epochs, but generator/branch updates are 2000/2600 versus 9500/13600 for N300, changing both sample size and compute budget. W1/W2 GIFT-Lite belongs to the N300 protocol with source key `GIFT-N50`; S1–S3 GIFT-Lite belongs to an independent N1000 protocol with separate schedules and validation-based selection, so weights and budgets are not interchangeable.

GIFT uses supervision horizons of 0.08/0.48 seconds and internal RK4 dt = 0.02; W1 advances 150 internal steps and saves every 0.04 seconds for 76 points. ANIE uses its best validation-MSE model from epoch29; training stopped at epoch1029 under the original early-stopping rule. ANIE participates only in W1.

| Confirmation metric | GIFT | GIFT-Lite |
| --- | ---: | ---: |
| W1 future75 mean relative L2 | 0.024687 | 0.036608 |
| W1 t8 relative L2 | 0.038113 | 0.065787 |
| W2 N64 future25 mean relative L2 | 0.021098 | 0.026436 |
| W2 N96 future25 mean relative L2 | 0.020633 | 0.024404 |
| W2 N128 future25 mean relative L2 | 0.020610 | 0.024372 |

Relative L2 values are fractions: 0.01 means 1%. All seven W1 methods and six W2 methods have 0/180 failures on their respective tasks in both populations. Conclusions reflect single-model performance under fixed recipes; they do not isolate architecture, represent the best achievable performance after extensive tuning, or estimate training randomness. Nonfinite predictions and solver rejections retain the corresponding trajectories and reasons; undefined full-population metrics are recorded as null, with finite-only results used solely for diagnosis.

## Numerical and figure files

Current W3 results cover 50 noise-free parameter combinations, one ID0 trajectory per combination, 250 slots across five methods and 750 parameter-readout rows. See the [W3 protocol](W3_PROTOCOL.md) and [result manifest](../results/formal/W3_single_trajectory_20261010/published.json) for complete results and the locked protocol.

| Experiment | Result directory and scope |
| --- | --- |
| W1 | [W1 result package](../results/formal/W1_flow_prediction_20261007/): two populations of 180 trajectories each, future75, t8, MSE and failures; confirmation and exploration saved separately. |
| W2 | [W2 result package](../results/formal/W2_cross_resolution_20261007/): two populations, three grids, future25, t6, MSE, frequency-band metrics and failures; excludes ANIE. |
| W3 | [W3 result package](../results/formal/W3_single_trajectory_20261010/): 50 noise-free parameter combinations, 250 method/case slots, 750 finite parameter-readout rows; per-case values, summary tables and [Figure 4a–c parameter-specific absolute percentage errors](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg). |
| S1–S3 | Numerical results for their independent supplementary protocols in `S1_high_frequency_branch/`, `S2_recursive_local_correction/` and `S3_initial_distribution/`, respectively. |

W1/W2 scientific result packages contain CSV summaries, editable SVGs, curve CSVs, selected field arrays `source_fields.npz` and provenance manifests. Tables retain actual model keys while figures use paper method names; task identifiers are consistently W1/W2. PNG field panels reside in `assets/figures/`; presentation and color scales are detailed in [FIGURES.md](FIGURES.md). Manifests record source numerical reports, model/data identities and file SHA-256. Selected field arrays provide figure evidence, not complete training or prediction datasets.

W1 panels show vorticity fields on the left and signed errors prediction−truth on the right, at fixed t5/6/7/8 and IDs100005/1045. W2 N128 panels at t6 show fields above errors, with fixed IDs100110/1150. All fields use native vector grids; fields and errors each share symmetric scales across all methods, without clipping finite values or selecting trajectories by error. Curves retain W1 PCHIP through seven report points and W2 logarithmic axes across three grids; main-table metrics use all future75/future25 nodes.

ANIE training and numerical audits for both W1 populations are bound to `prediction_aligned`; provenance for other models and W2 appears in their result manifests.

W1/W2 weights, complete raw HDF files and optimizer/random states remain in declared external experiment directories and are not packaged with Git. Result packages support reading tables and viewing figures but cannot restore training. Reinference requires the corresponding weights and data; retraining requires full training configurations. Environments and commands are in [SETUP.md](SETUP.md) and [EXPERIMENT_EXECUTION.md](EXPERIMENT_EXECUTION.md).

```shell
python -m scripts.verify_published_results --experiment W1 --result-dir results/formal/W1_flow_prediction_20261007
python -m scripts.verify_published_results --experiment W2 --result-dir results/formal/W2_cross_resolution_20261007
```

Integrity commands check file bytes and provenance bindings without training or recomputing predictions; different hardware, mathematical libraries and upstream random operators may change floating-point results.

## Independent parameter identification and supplementary protocols

W3 uses independent parameter cases and one local ID0 trajectory per case, without observation noise. Each of 50 truths has one trajectory; five methods each complete 50 runs, yielding 750 parameter-readout rows. Mean/median APE, complete coefficients and timings are in the [experiment record](../EXPERIMENTS.md#43-w3-identification-across-50-noiseless-parameter-cases) and [result package](../results/formal/W3_single_trajectory_20261010/README.md). Full open-library supports are reported separately; all 100 PINN L-BFGS runs reached the function/gradient evaluation limit. GIFT has the lowest mean APE for ν and β, while PDE-FIND is lowest for γ. Finite readouts do not imply accurate parameters; GIFT's maximum ν APE is 59.32%. Figures 4a–c show absolute percentage error distributions for ν, β and γ, with APE=100×|estimate−truth|/|truth| and all 50 actual readouts per method/parameter retained. Solid Tukey boxes show Q1–Q3 of individual APEs; extended lines through boxes and row Median values represent medians of 50 APEs. With IQR=Q3−Q1, thin whiskers and short caps extend to the outermost actual observations within inclusive fences [Q1−1.5×IQR, Q3+1.5×IQR]. All actual observations, including those beyond whiskers, remain without duplicate flier symbols or separate mean markers. These are empirical distributions across parameter cases, not confidence intervals. Dots are deterministically arranged on seven equally spaced lanes without changing their APE coordinates. Dots have nominal diameter 2.1 pt and adjacent lanes are separated by 2.7 pt for a compact point layer. Each real point comprises 12 concentric, stroke-free, same-color transparent discs, with a radial Gaussian alpha profile for soft edges (sigma 0.42 relative to radius; nominal central alpha 0.95). This softens individual real markers without KDE, data smoothing or artificial observations. Method names are bold black; dots, solid boxes and whiskers use saturated dark method colors. Median lines exactly match box colors, with width 1.6 pt and length about 2.1 times box height; axes are solid black. Horizontal ticks point upward into the plot, numerical labels remain below the axis, and the axis extends beyond the final tick by 3% of its current span without changing data coordinates or tick positions. Plot text is limited to method names, coordinate ticks, parameter-specific horizontal-axis labels and each method's numerical Median value. Each figure has one full-range panel; all five methods share a linear axis starting at zero, [0,300]% for ν and β and [0,110]% for γ, without clipping finite values. [Figure 4a: ν SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg) ([PNG](../assets/figures/w3_parameter_nu_errors.png)) · [Figure 4b: β SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_beta_errors.svg) ([PNG](../assets/figures/w3_parameter_beta_errors.png)) · [Figure 4c: γ SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_gamma_errors.svg) ([PNG](../assets/figures/w3_parameter_gamma_errors.png)); provenance and rendering rules are in the [figure manifest](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_manifest.json).

S1–S2 use N1000 GIFT and GIFT-Lite models: GIFT's generator and branch each train for 500 epochs and use terminal states; GIFT-Lite uses 50 training trajectories, separate schedules and validation-based selection. Training IDs are 0–999 and test IDs1040–1219; GIFT-Lite trains on IDs0–49. These model definitions and statistical scopes are stated separately from the W1/W2 single-model protocol.

In S1, enabling the branch reduces full-field and Q21-state prediction errors but worsens the per-state mean instantaneous Q21 derivative error; small denominators and long tails must be interpreted alongside quantiles. In S2, N1000 GIFT never triggers local correction and remains finite with correction on/off; GIFT-Lite without correction destabilizes on trajectory1062 at t6.74, after which finite-only means use 179 trajectories and cannot establish an overall accuracy advantage.

S3 uses an independent N1000 training protocol to compare four-vortex and smooth Gaussian random-field distributions. Baselines receive 46 historical frames from t4.1–5.0; GIFT receives a single t5 state and predicts 150 steps to t8. Each distribution is trained and tested separately, making this a within-distribution comparison. Gaussian training IDs are 1220–2219 and test IDs2260–2439; the four-vortex control uses its corresponding N1000 models, IDs1040–1219 and fixed example1045. This control is not an N300 W1 measurement.

In S3, GIFT's t8 mean is 0.021035 and the closest U-NO value is 0.173805; GIFT, U-NO, FNO-2D, FNO-3D and U-Net remain finite on 180/180 trajectories. GIFT-Lite has 17/180 trajectories destabilize after approximately t6.9, so t7 and later statistics cover finite subsets only and its curve stops at t6.5. Full failure counts accompany results; finite-only values cannot be ranked directly against other methods' full-population values, and this experiment does not support conclusions for arbitrary initial-condition distributions.
