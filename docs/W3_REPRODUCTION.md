# Current W3 preparation, training, measurement and aggregation

The current protocol is `single_trajectory_terminal_v1`: 50 fixed parameter cases, each retaining only training ID0, with N64 observations comprising 501 frames over t0–10 at dt0.02 and no independent validation trajectory. GIFT uses 6000+6000 updates and the fixed terminal checkpoint of each stage. PINN retains the original seed1234, 24,000 training measurements, 6,000 held-out measurements, 84,000 physics points and complete original optimization budget. Parameter truths are used only for data QA and post-measurement APE, never for training initialization or labels.

Run the following commands from the repository root. Replace the example variables with actual installation paths; the two Python executables may point respectively to a PyTorch environment and a PINN environment with the required native TensorFlow runtime dependencies. Prepare external algorithms and runtime interfaces outside the GIFT repository as described in the [manual adaptation instructions](EXTERNAL_MANUAL_ADAPTATION.md). GIFT's own training and data generation require no external algorithm implementation; PDE-FIND and PINN require their corresponding external interfaces and upstream source.

```powershell
$TorchPython = 'C:\envs\gift\python.exe'
$PinnPython = 'C:\envs\gift-pinn\python.exe'
$Dataset = 'D:\datasets\GIFT-dataset'
$Run = 'D:\runs\w3_current_fresh'
$env:GIFT_EXTERNAL_ROOT = 'D:\upstreams'
$env:GIFT_EXTERNAL_ADAPTER_ROOT = 'D:\gift-external-runtime'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

## Using the ScienceDB data package

First verify the downloaded package completely, then create the current protocol and input gate for a new run directory:

```powershell
& $TorchPython -B -m scripts.verify_data --root $Dataset --full-array-scan
& $TorchPython -B -m scripts.prepare_w3_run --root $Run --dataset-root $Dataset
& $TorchPython -B -m scripts.prepare_w3_run --root $Run --dataset-root $Dataset --execute
```

The first preparation command displays only a plan. With `--execute`, it rechecks the identities of observations, receipts, locks, QA and cohorts for all 50 cases, scans every observation for finite values, initial-condition parameters, IDs and times, recomputes the original FD data-quality gate, and reconstructs all sensors, times, training/holdout splits and LHS designs with seed1234. It recomputes `[u,v,ω]` sampling targets from each case's actual stored fields. The ScienceDB package is read-only. Float32 machine differences in initial conditions/FD use the existing data-QA tolerances; the cross-FFT-library target check uses `rtol=5e-6, atol=5e-7`, with actual differences written to new records.

After every check passes, the new run receives `PROTOCOL_LOCK.json`, `PREPARATION_SOURCES.json`, `W3_PREPARATION.json`, `data/DATA_COHORT_QA.json` and `FULL_INPUT_GATE.json`. Each successful recheck is saved separately under `preparation_checks/`, retaining its FD/target floating-point diagnostics. Machine-rounding differences for identical inputs within the permitted tolerances neither overwrite the first preparation nor prevent a recheck. Source hashes are recorded as provenance; ordinary runs need not be byte-identical to historical source. Historical data receipts remain unchanged, and the gate never treats an incomplete case subset as all 50 cases.

## Single-case execution and recovery

The example below uses `parameter_000`. Omit `--execute` first to inspect the plan. The GIFT command selects the one-trajectory, zero-validation, terminal configuration from the current case receipt.

```powershell
& $TorchPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method GIFT --action train --device cuda --execute
& $TorchPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method GIFT --action measure --device cpu --execute

& $TorchPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PDE-FIND --action measure --device cpu --execute
& $TorchPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PDE-FIND-KC --action measure --device cpu --execute

& $PinnPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PINN-SR --action train --device gpu --execute
& $PinnPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PINN-SR --action measure --device cpu --execute
& $PinnPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PINN-SR-KC --action train --device gpu --execute
& $PinnPython -B -m scripts.run_w3_parameter_job --root $Run --dataset-root $Dataset --case parameter_000 --method PINN-SR-KC --action measure --device cpu --execute
```

Replace the case with `parameter_000`–`parameter_049` in turn to complete all 250 method/instance slots. Each case/method must use an independent native journal. After interruption, add `--resume` to the original training command only if that slot's native journal exists; completed training/measurements can be read and verified by the original command without refitting. Recorded scientific failures, including OOM or non-finite values, remain failures of their fixed slots: they cannot be deleted and replaced by a resampled case or presented as fresh initialization of the same run. Infrastructure interruptions retain attempt logs and recover from their own saved state. `--claim-token` may record a scheduler assignment identifier; manual single-machine execution does not require the historical two-machine scheduling lock.

New training and measurements write only to `$Run/training`, `$Run/jobs` and `$Run/job_attempts`. They do not read published results to fill missing models or measurements.

## Regenerating current observations from initial conditions

Independent generation uses a new directory, for example `$GeneratedRun = 'D:\runs\w3_current_regenerated'`. Keep the downloaded package and old run unchanged.

```powershell
$GeneratedRun = 'D:\runs\w3_current_regenerated'
& $TorchPython -B -m scripts.generate_w3_parameter_data --root $GeneratedRun --parameter-csv "$Dataset\identification\w3\parameters.csv" --profile single_trajectory_terminal_v1 --device cuda --batch-size 1 --execute
```

This entry point directly reintegrates all 50 equations from the fixed ID0 four-vortex initial condition, with one training trajectory and zero validation trajectories per case. It outputs the current schema and explicitly records `new_integration_performed=true`. It does not misrepresent new integration as bitwise-lossless extraction of old data. The original `validation_best_v1` profile remains available, and omitting `--profile` preserves the historical 1 train + 20 validation behavior. Results from the historical profile cannot be treated as the current main experiment.

After generation, prepare sampling for each case and then create the complete gate:

```powershell
0..49 | ForEach-Object {
    $Case = 'parameter_{0:d3}' -f $_
    & $TorchPython -B -m scripts.run_w3_parameter_job --root $GeneratedRun --case $Case --action prepare-sampling --execute
    if ($LASTEXITCODE -ne 0) { throw "Sampling failed for $Case" }
}
& $TorchPython -B -m scripts.prepare_w3_run --root $GeneratedRun --execute
```

Then reuse the preceding section's training/measurement commands, replacing `--root` with `$GeneratedRun` and removing `--dataset-root $Dataset`. The generator saves the fixed parameter CSV, initial conditions and cohort QA locally in the run. Newly generated data may differ from the package's historical data at float32 precision across FFT libraries/devices; first inspect difference magnitudes and their accumulation during evolution before deciding whether a scientific anomaly exists. Retraining does not promise bitwise reproduction of existing weights or every historical estimate.

## Current 50×5 aggregation

Current aggregation uses a dedicated entry point, rather than the old 15-slot aggregator for three noise conditions:

```powershell
& $TorchPython -B -m scripts.aggregate_w3_current --root $Run --output "$Run\summary_current" --execute
```

It verifies preparation/protocol/cohort and the three numerical files for every completed measurement. Neural methods must also have the actual model, training report, case binding and native completion from the same run: measured hashes of GIFT weights and reports must match the manifest, and PINN results must agree with both `COMPLETED.json` and the actual `terminal_state.npz`. Merely declaring a checkpoint hash cannot substitute for these files. Outputs include `parameter_cases.csv`, `parameter_results.csv`, `method_metrics.csv`, `numeric_report.json` with provenance and costs, and a hash commit. All 250 slots and 750 parameter rows are retained. Scientific failures are `failed`; unfinished or recoverable interruptions are `pending`, and published data are never used to fill gaps. Any pending slot yields `incomplete`; terminal states for all slots with some failures yield `complete_with_failures`. If a method has not completed all 50 cases, its formal mean/median/min/max remain blank; successful-subset means appear only in columns named with `diagnostic_available_case`.

Use a new output directory for each aggregation. Aggregation retains each slot's recorded training costs, measurement times and all attempts from its own run, including failures and interruptions; missing costs are not estimated. Aggregation performs no training or reintegration and changes no historical experimental values. New figures may be drawn separately from the verified CSV files.
