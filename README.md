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
experiment1_analysis.py    # Spectral aging replication and sex differences analysis
experiment2_classification.py  # PD vs control LODO classification, confound controls,
                               # and sensitivity analyses (frontal removal, gamma removal,
                               # MoCA residualization, label permutation)
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
