# GIFT dataset

This data package collects observation inputs used by current W1, W2, W3, S1, S2 and S3, together with numerical source data for formal results. Field-array axes are consistently documented as `[trajectory,time,y,x]`. Model weights, training recovery states, external algorithm source and complete prediction fields are excluded.

## Directories and purposes

| Directory or file | Contents |
| --- | --- |
| `prediction/four_vortex/` | Base observations with four-vortex initial conditions; used by independent N1000 supplementary protocols, GIFT-Lite and S1/S2, and as the parent data for W1/W2 training selection |
| `prediction/w1_w2/` | N300/N50 training slices and N64/N96/N128 confirmation truths for current W1/W2 |
| `identification/w3/` | 50 noise-free parameter combinations for current W3, one ID0 trajectory per combination and corresponding PINN samples |
| `s3_gaussian/` | Gaussian observations under the current `gaussian_s3` identity and original generation records |
| `source_data/W1/` through `source_data/S3/` | CSV, JSON and necessary NPZ files for six formal experiments, including per-case results, summaries and figure source data |
| `metadata/` | Array structures, sample splits, provenance and experiment-coverage tables; original W1/W2 locks reside in `metadata/provenance/w1_w2/` |
| `dataset.json` | `gift.dataset.v1` path-role table for code to locate current files |
| `manifest.json` | File-size and SHA-256 manifest for the current package |

File fields are described in `DATA_DICTIONARY.md`; machine-readable definitions appear in `metadata/schema.json`, `metadata/splits.json`, `metadata/provenance.json` and `metadata/experiment_coverage.json`. Original result or figure manifests retained under `source_data` record historical provenance; the root `manifest.json` defines current package members.

## Sample splits

Current W1/W2 select 300 trajectories in a fixed random order from base training-pool IDs 0–999; validation contains 20 trajectories, IDs 1000–1019. GIFT-Lite uses the first 50 entries of this ordered list, not base-pool IDs 0–49. Training observations comprise 76 true nodes over t=5–8 at dt=0.04.

Confirmation IDs 100000–100179 and exploration IDs 1040–1219 each contain 180 trajectories. W1 evaluates 75 future nodes over t=5–8 and W2 evaluates 25 future nodes over t=5–6 on N64/N96/N128, both excluding the input initial state. Confirmation is independent of training and validation, but its scores have been viewed and do not constitute a new blind test.

Independent N1000 supplementary protocols use training IDs 0–999, validation IDs 1000–1039 and test IDs 1040–1219; full-time-window validation uses 1000–1019, while `validation_aux` uses 1020–1039. Their GIFT-Lite uses base training-pool IDs 0–49. The same ID on different grids denotes the same physical initial condition; validation and test data do not enter training.

Each W3 case `parameter_000`–`parameter_049` contains one noise-free trajectory for its equation parameters, with local ID0 and no external validation. These 50 trajectories do not use global prediction-experiment IDs. The old 0%/1%/5% noise protocol is not part of current W3.

## Usage and versions

Point `GIFT_DATA_ROOT` to the package root. The current local package is `../GIFT-dataset`; a complete legacy backup is adjacent `GIFT-dataset_legacy_20261020`. Run from the GIFT project root:

```shell
python -m scripts.verify_data --root /absolute/path/GIFT-dataset
```

The current package locates inputs through its role table; legacy packages without `dataset.json` still use the old layout. Commands with explicit HDF5 paths should use current paths. Data regeneration and model training are separate steps; see project files `docs/DATA_GENERATION.md` and `docs/SETUP.md`.

Data is licensed under CC BY 4.0; see `LICENSE.txt`. The existing ScienceDB record is *GIFT Navier–Stokes input data*, with review entry point `https://www.scidb.cn/en/anonymous/YnVVUjNl`. This reorganization produced only a local version and has not uploaded or updated the ScienceDB record; check the actual record version used when citing.
