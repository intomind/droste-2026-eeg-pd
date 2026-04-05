# Droste (2026) — Cross-laboratory EEG spectral biomarkers of Parkinson's disease

Analysis code for:

> **A harmonized corpus of 8,974 EEG subjects from 195 studies validates Parkinson's spectral biomarkers across seven independent laboratories without artifact rejection**
>
> Kristian Droste, IntoMind, Inc.
>
> Preprint: [bioRxiv link forthcoming]

## Overview

This repository contains the analysis scripts used to produce the results in the above manuscript. The study assembled 195 human EEG datasets (8,974 subjects, 5,041 hours) from OpenNeuro, applied a minimal preprocessing pipeline without artifact rejection, and demonstrated cross-laboratory generalization of Parkinson's disease spectral biomarkers using leave-one-dataset-out (LODO) classification across seven independent recording sites.

## Contents

```
experiment1_lifespan.py        # Spectral aging, sex differences, alpha asymmetry
experiment2_classification.py  # PD vs control LODO classification, confound controls,
                               # and sensitivity analyses
```

### experiment1_lifespan.py

Spectral analysis across the lifespan (all 8,974 subjects):

- Age–spectral correlation heatmap (channel x band, Bonferroni-corrected)
- Posterior alpha power vs age (scatter + LOESS, per-dataset coloring)
- Peak alpha frequency vs age
- Per-dataset consistency forest plot
- Sex differences heatmap (Cohen's d with bootstrap CIs)
- Hemispheric alpha asymmetry and BDI depression correlation

### experiment2_classification.py

Cross-dataset Parkinson's classification (743 subjects, 7 datasets):

- Leave-one-dataset-out (LODO) cross-validation
- Stratified 5-fold CV (upper-bound baseline)
- Age confound control (with/without age as feature)
- Dataset confound control (dataset identity classifier + in-fold residualization)
- Feature importance (logistic regression coefficients, random forest MDI)
- ROC curves and confusion matrices

With `--sensitivity` flag, also runs:

- Frontal channel removal (drop Fp1, Fp2, Fz — tests ocular artifact contribution)
- Gamma-band removal (drop gamma features — tests EMG contamination contribution)
- MoCA cognitive covariate analysis (spectral-only vs spectral+MoCA vs MoCA-residualized)
- Label permutation test (100 within-dataset permutations)

## Usage

```bash
# Experiment 1: spectral aging and sex differences
python experiment1_lifespan.py path/to/features.csv

# Experiment 2: PD classification (core analyses only)
python experiment2_classification.py path/to/features.csv

# Experiment 2: including sensitivity analyses
python experiment2_classification.py path/to/features.csv --sensitivity

# Custom output directory
python experiment2_classification.py path/to/features.csv -o path/to/output/

# MoCA analysis requires catalog.db (auto-detected or specify manually)
python experiment2_classification.py path/to/features.csv --sensitivity --catalog-db path/to/catalog.db
```

## Data

Source EEG datasets are publicly available on [OpenNeuro](https://openneuro.org/) under CC0 licenses. The full list of 195 dataset IDs used in this study is provided in the manuscript's SI Appendix, Table S1.

## Requirements

- Python 3.10+
- numpy
- pandas
- scipy
- scikit-learn
- statsmodels
- matplotlib

## Preprocessing

The preprocessing pipeline that produced the input feature matrix is described in detail in the manuscript's Materials and Methods section. Briefly: raw EEG was mapped to 19 standard 10–20 electrode positions, resampled to 500 Hz, re-referenced to common average, notch filtered at 50 and 60 Hz, high-pass filtered at 0.5 Hz (zero-phase FIR), z-score normalized per channel, and segmented into non-overlapping 4-second windows. No ICA, artifact rejection, or bad channel interpolation was applied. Spectral features (relative band power for delta, theta, alpha, beta, gamma; peak alpha frequency) were extracted via Welch's method and averaged per subject.

## Citation

If you use this code, please cite:

```
Droste, K. (2026). A harmonized corpus of 8,974 EEG subjects from 195 studies validates
Parkinson's spectral biomarkers across seven independent laboratories without artifact
rejection. [Preprint]. https://doi.org/[forthcoming]
```

## Contact

Kristian Droste — kris@intomind.com

ORCID: [0009-0008-4727-8962](https://orcid.org/0009-0008-4727-8962)

## License

MIT
