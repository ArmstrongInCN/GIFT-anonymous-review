# Data dictionary

## Common fields

| Field | Type, axes and meaning |
| --- | --- |
| `vorticity` | float32, `[trajectory,time,y,x]`, two-dimensional vorticity field |
| `time` | float64, `[time]`; absolute times or file-declared relative times |
| `absolute_time` | float64 absolute times for W1/W2 training slices and confirmation truth, `[time]` |
| `trajectory_index` | in t64, `[trajectory]`, trajectory IDs under the file's protocol |

The spatial domain is periodic `[0,2π)²`, without duplicate endpoints; spatial axes are ordered y, x. No SI conversion is specified. NaN and Inf do not denote missing data; observation arrays must be finite. Array group names, shapes and dtypes are in `metadata/schema.json`; trajectory lists and saved-time summaries are in `metadata/splits.json`. Complete time axes and original attributes remain in each HDF5 file.

## Base four-vortex observations

Paths in the table below are prefixed by `prediction/four_vortex/`. Base training IDs are 0–999, validation IDs1000–1039 and test IDs1040–1219; `validation_aux` denotes 1020–1039 rather than additional test trajectories. GIFT-Lite in the independent auxiliary protocol uses IDs0–49.

| File | Observation contents |
| --- | --- |
| `train1000_n64_t0_t10_dt0p02.h5` | 1,000 training and 20 complete validation trajectories; N64, t=0–10, dt=0.02 |
| `train1000_n64_t0_t10_dt0p1.h5` | The same training and complete validation trajectories; N64, t=0–10, dt=0.1 |
| `standard_n64.h5` | 50 training and 20 complete validation trajectories; observations for 180 test and 20 auxiliary validation trajectories over t=5–10, dt=0.5 |
| `test_n64_t4p1_t6_dt0p1.h5` | 180 test and 20 auxiliary validation trajectories; N64, t=4.1–6, dt=0.1 |
| `test_n96_n128_t4p1_t6_dt0p1.h5` | Paired N96/N128 test and auxiliary validation trajectories; t=4.1–6, dt=0.1 |
| `test_n64_n96_n128_dt0p02.h5` | Paired test and auxiliary validation trajectories; N64 spans t=4.1–8, N96/N128 span t=4.1–6, dt=0.02 |

Current containers use canonical trajectory numbering. Renumbering and directory reorganization do not interpolate, smooth, scale, round or reintegrate observations. `metadata/initial_conditions_extra950.json` stores four-vortex initial-condition parameters and integration settings for IDs50–999; each initial condition comprises four rows of `[circulation,sigma,cx,cy]` for independent regeneration.

## Current W1/W2

The following paths are prefixed by `prediction/w1_w2/`. The N300 list is `default_rng(2026100304).permutation(1000)[:300]`; N50 is the first 50 entries of that ordered list. Both slices use validation IDs1000–1019.

| File | Groups, shapes and saved times |
| --- | --- |
| `train300_validation20.h5` | `training/vorticity` is `[300,76,64,64]`, `validation/vorticity` is `[20,76,64,64]`; t=5–8, dt=0.04 |
| `train50_validation20.h5` | `training/vorticity` is `[50,76,64,64]`, `validation/vorticity` is `[20,76,64,64]`; t=5–8, dt=0.04 |
| `confirmation_n64.h5` | `test/vorticity` is `[180,151,64,64]`; IDs100000–100179, t=5–8, dt=0.02 |
| `confirmation_n96_n128.h5` | `N96/test/vorticity` and `N128/test/vorticity` are `[180,51,96,96]` and `[180,51,128,128]`, respectively; the same confirmation IDs, t=5–6, dt=0.02 |

Formal evaluation samples every other integer frame from dense truth, giving 76 nodes for W1 and 26 for W2, without temporal interpolation. In W1/W2 slice and confirmation containers, `absolute_time` covers the absolute intervals in the table, while `time` starts at zero (0–3 for training slices and N64 confirmation,0–1 for higher-grid confirmation). The exploratory population retains base-observation test IDs1040–1219. Weights and complete predicted fields for W1's seven and W2's six methods are not in this package. Original locks and data receipts reside in `metadata/provenance/w1_w2/`, whose historical absolute paths remain original provenance records.

## Current W3

`identification/w3/parameters.csv` stores the 50 fixed parameter truths, `INITIAL_CONDITIONS.json` the initial condition and `DATA_COHORT_QA.json` the cohort checks. Each of `parameter_000`–`parameter_049` retains:

- `standard_ns_n64_full_spectrum.h5`: one local ID0 trajectory, float32 `[1,501,64,64]`, t=0–10, dt=0.02.
- `DATA_RECEIPT.json`, `DATA_LOCK.json`, `DATA_QA.json`, `OBSERVATION_PRESERVATION.json`: truths, array identities and provenance bindings.
- `auxiliary/pinn_sampling/noise_000_seed1234.npz` and its `.receipt.json`: fixed PINN sampling for the case; `noise_000` is the retained noiseless-condition name in the sampling filename.

The equation is `ω_t = -β(uω_x + vω_y) + νΔω − 4γcos(4y)`. The ν range is [0.0075, 0.015], and β and γ range over [0.75, 1.25]; the parameter CSV specifies exact combinations. Each case has one training trajectory and no external validation; local ID0 trajectories from different cases must not be merged as one prediction trajectory.

PINN sampling per case contains 30,000 measured `(x,y,t)` positions and `(u,v,ω)` targets, split into 24,000/6,000 training/same-trajectory holdout rows. An additional 60,000 LHS positions bring the physics-point total to 84,000 (including the 24,000 training positions). Holdout rows are not additional trajectories. GIFT uses fixed terminal checkpoints; models and training states remain in separate run directories. The old 0%/1%/5% noise data are not part of the current package.

## S3 Gaussian initial conditions

`s3_gaussian/trajectories.h5` and `lite.h5` use the current `gaussian_s3` identity and retain original observation arrays: N64, t=0–10, dt=0.02,501 frames. `trajectories.h5` contains 1,000 training,20 validation,20 auxiliary validation and 180 test trajectories; `lite.h5` contains the first 50 trajectories of the matching training pool and 20 validation trajectories. Training IDs are 1220–2219, validation IDs2220–2259 and test IDs2260–2439. `COMPLETE.json`, `manifest.json` and `run.json` retain original generation records. This package's metadata specify complete fields and splits.

## Numerical source data and provenance

`source_data/{W1,W2,W3,S1,S2,S3}/` retains formal-result CSV, JSON and necessary NPZ files for numerical verification and replotting. W3 retains only the source data for the current parameter-distribution figures; the old joint-parameter figure is not the current Figure 4. The package contains no rendered SVG/PNG figures, complete prediction fields or training-recovery states. Original result/plotting manifests still describe their original artifacts; the root `manifest.json` defines this release's file membership and hashes.

Base four-vortex observations satisfy `ω_t + u·∇ω = 0.01Δω − 4cos(4y)`. Original generation uses full-spectrum Fourier pseudospectral ETDRK4, internal dt=0.005 and padding99/147/195 for N64/N96/N128, without truncating state spectral modes. Data-directory reorganization changes neither these numerical conventions nor historical scientific results.
