# RM-MDE: Reliable Multilevel Image Thresholding

This repository contains the reproducibility package for the manuscript **A criterion-agnostic memetic differential evolution for reliable multilevel image thresholding**.

The repository is being populated from the locally validated package. The scientific results are frozen; later changes should be limited to metadata, documentation, or submission requirements.

## Main contents
- `source_code/` — validated analysis and reproduction scripts.
- `data_results/` — locked numerical results and indices.
- `figures/` — figures used in the manuscript.
- `supplementary/` — supplementary analyses, including the full PSO/DE baseline-sensitivity grid.
- `review_response/` — reviewer-response support files.
- `ISSA_legacy/` — the earlier ISSA implementation retained only for audit; it is not used by the final pipeline.

## Main methodological points
- Certified optima: exhaustive search for 2–3 thresholds and an exact dynamic program for 4–8 thresholds.
- RM-MDE uses differential evolution with exact coordinate-wise local refinement.
- Main RM-MDE setting: NP=20, G=60, F=0.5, Cr=0.9, p_ls=0.15, local-search cap of 4 sweeps.
- SDE baseline: F=Cr=0.25 fixed, evaluated under the common study budget.
- ISSA is a corrected one-dimensional adaptation of the published method, not a reproduction of its original 2-D experimental protocol.

## Software environment
Results were generated with Python, NumPy 2.2.6, SciPy 1.16.1, scikit-image 0.25.2, and Matplotlib 3.10.5 in single-threaded execution.

## Reproducibility note
The long benchmark experiments were executed and validated before packaging. Representative reproduction checks from the packaged location were passed, including exact-DP validation, statistics regeneration, ablation/time-matched checks, figure regeneration, and the ISSA audit. The regenerated 120-instance statistics match the archived numerical values exactly (120/120 instances; numerical difference 0.0).

## Data availability
The repository is intended to provide the code and derived results required to reproduce the analyses reported in the manuscript. Large raw image datasets are not redistributed here when they are already available through the cited public source.

## License
A license has not yet been selected. Until a license file is added, no additional permission to reuse the code should be assumed.
