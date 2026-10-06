# Supplementary material: baseline-parameter robustness check (PSO and DE)

Protocol (fixed before the runs; script `03_source_code/generalized_code/bundle2/baseline_sensitivity.py`): 8 images x 3 criteria x n = 6, 7, 8 x seeds 0-9 = 720 runs per configuration; reference = DP-certified optimum; success = relative gap < 1e-8. RM-MDE was not re-run: its results are the stored ones of `time_matched_DE_PSO.csv` for the same instances and seeds (success 90.1%, mean gap 1.7e-5). 'RM-MDE better / tie / worse' counts the paired runs in which the RM-MDE gap is strictly smaller than / within 1e-8 of / strictly larger than the baseline gap.

Per-run results: `05_data_results/baseline_sensitivity_runs.csv`; summary: `05_data_results/baseline_sensitivity.json`. The 16 equal-generation entries are baseline configurations or checks, not all parameter settings of the same kind (9 DE (F, Cr) combinations, 3 PSO variants, 2 + 2 population-size checks).

## Part 1: equal generations (NP = 20, G = 60 unless stated)

| Configuration | Success (%) | Mean gap | Median gap | Otsu | Kapur | MCET | n=6 | n=7 | n=8 | RM-MDE better / tie / worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DE F=0.3 Cr=0.2 | 0.8 | 2.89e-03 | 6.4e-05 | 0.4 | 1.7 | 0.4 | 0.8 | 1.7 | 0.0 | 707 / 6 / 7 |
| DE F=0.3 Cr=0.5 | 11.5 | 2.36e-03 | 2.2e-05 | 10.8 | 17.9 | 5.8 | 22.9 | 8.3 | 3.3 | 629 / 80 / 11 |
| DE F=0.3 Cr=0.9 | 4.3 | 2.25e-03 | 4.7e-05 | 3.3 | 5.4 | 4.2 | 9.6 | 1.2 | 2.1 | 686 / 32 / 2 |
| DE F=0.5 Cr=0.2 | 0.0 | 2.24e-03 | 6.6e-05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 713 / 0 / 7 |
| DE F=0.5 Cr=0.5 | 4.0 | 1.18e-03 | 2.1e-05 | 2.5 | 7.9 | 1.7 | 8.8 | 3.3 | 0.0 | 684 / 28 / 8 |
| DE F=0.5 Cr=0.9 | 19.0 | 9.42e-04 | 3.1e-06 | 19.6 | 20.0 | 17.5 | 37.5 | 15.4 | 4.2 | 574 / 131 / 15 |
| DE F=0.8 Cr=0.2 | 0.0 | 1.60e-03 | 8.4e-05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 716 / 0 / 4 |
| DE F=0.8 Cr=0.5 | 0.0 | 1.68e-03 | 7.7e-05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 714 / 0 / 6 |
| DE F=0.8 Cr=0.9 | 0.0 | 1.54e-03 | 6.1e-05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 715 / 0 / 5 |
| PSO ring-constriction | 0.6 | 1.20e-03 | 2.7e-05 | 0.8 | 0.0 | 0.8 | 1.7 | 0.0 | 0.0 | 710 / 3 / 7 |
| PSO gbest-constriction | 12.1 | 1.84e-03 | 8.5e-06 | 10.8 | 12.1 | 13.3 | 25.4 | 8.3 | 2.5 | 628 / 81 / 11 |
| PSO ring-inertia | 0.6 | 1.05e-03 | 3.7e-05 | 0.4 | 0.4 | 0.8 | 1.7 | 0.0 | 0.0 | 712 / 3 / 5 |
| DE NP=40 G=30 | 0.0 | 1.22e-03 | 5.3e-05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 714 / 0 / 6 |
| PSO NP=40 G=30 | 0.0 | 1.78e-03 | 1.1e-04 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 716 / 0 / 4 |
| DE NP=10 G=120 | 1.0 | 3.00e-03 | 1.2e-04 | 0.8 | 0.8 | 1.2 | 2.5 | 0.4 | 0.0 | 709 / 8 / 3 |
| PSO NP=10 G=120 | 9.9 | 1.52e-03 | 1.4e-05 | 10.0 | 14.6 | 5.0 | 22.1 | 5.8 | 1.7 | 643 / 70 / 7 |

Population-size checks use the same number of evaluations (1200): NP = 40 with G = 30, and NP = 10 with G = 120 (DE: F = 0.5, Cr = 0.9; PSO: ring + constriction).

## Part 2: seed-matched wall-clock budget (1x of the RM-MDE time for the same instance and seed)

Run for the DE configuration and the PSO variant with the highest success in Part 1. The ring + constriction PSO (main setting) and the DE main setting at the matched budget are also reported in Section 5.4 of the manuscript (PSO 48.3%, DE 32.6%).

| Configuration | Success (%) | Mean gap | Median gap | RM-MDE better / tie / worse |
|---|---:|---:|---:|---|
| DE F=0.5 Cr=0.9 | 32.6 | 4.51e-04 | 5.1e-07 | 473 / 224 / 23 |
| PSO gbest-constriction | 42.1 | 1.51e-03 | 2.8e-07 | 406 / 288 / 26 |

## Limits of this check
- The grids are small; velocity clamping, other DE mutation strategies and self-adaptive DE were not tried.
- The ring + inertia PSO variant and the population-size checks were not run at matched time.
- The same eight images are used for every configuration; no inferential test is applied here.
- The check shows that, within the tested ranges, the comparison does not change; it does not show that no better-tuned baseline exists.
