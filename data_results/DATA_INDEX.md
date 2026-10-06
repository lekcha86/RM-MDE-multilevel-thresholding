# Index of result files

Status: **final** = used by the revised manuscript and the response letter; **intermediate** = produced on the way and still used; **superseded** = first-version file, kept for traceability only, not used for any reported number.

| File(s) | Content | Status |
|---|---|---|
| `exact_optima_dp.json` | DP-certified optimal value and threshold vector (bin indices) for 3 criteria x 8 images x n = 4-8 (MCET with x = i) | final (reference of every success rate) |
| `exact_optima_mcet_gray0.json`, `mcet_offset_validation.json/.csv`, `mcet_offset_rmmde_comparison.json` | MCET with x = i - 1: optima, comparison with x = i, RM-MDE success under both conventions | final |
| `main_rerun_table2.json` | Table 2 (3 images, n = 2-8, 15 seeds): success and mean time of PSO, DE, SDE (F = Cr = 0.25), corrected ISSA, RM-MDE | final |
| `main_rerun_conv.json` | mean relative gap per generation (MCET, Cameraman, n = 5, 15 seeds) | final |
| `main_rerun_quality.json` | PSNR and SSIM per image and method (MCET, n = 5, seeds 0-9) | final |
| `stats_otsu.json`, `stats_kapur.json`, `stats_mcet.json` | 120-instance study of the first version | PSO, DE and RM-MDE (`MDE`) entries: final (regenerated identically by `regen_stats120.py`); **SDE and ISSA entries: superseded** |
| `regen_stats120.json` | regeneration of the PSO / DE / RM-MDE entries and comparison with the stored ones (120 of 120 identical) | final |
| `sde_rerun_stats120.json` | 120-instance study with SDE (F = Cr = 0.25) and corrected ISSA: per-instance values, Friedman / Wilcoxon summaries (`summary_new_main`) | final |
| `sde_rerun_table2.json` | SDE (F = Cr = 0.25) in the Table-2 design | intermediate (its values are included in `main_rerun_table2.json`, which is the table used) |
| `issa_rerun_stats120.json` | 120-instance study, corrected ISSA only (summary with the first-version SDE is an intermediate) | intermediate (its ISSA instance values are used) |
| `issa_rerun_table2.json` | ISSA at G = 60 and G = 30 (Table-2 design) | intermediate (the G = 30 values are the sensitivity result) |
| `issa_rerun_quality.json` | PSNR / SSIM of corrected ISSA | superseded by `main_rerun_quality.json` |
| `quality_mcet_n5.json` | PSNR / SSIM / **FSIM** of the first version | **superseded** (FSIM not reproducible; ISSA and SDE values replaced) |
| `issa_levy_check.json` | corrected ISSA with the standard or the literal reading of eq. (10) of Wu and Yuan | final (sensitivity) |
| `sde_sensitivity.json`, `sde_sensitivity_runs.csv` | SDE variants (randomized F, fixed F, Ali et al. setting) against DE, PSO, RM-MDE (720 runs each) | final (sensitivity) |
| `baseline_sensitivity.json`, `baseline_sensitivity_runs.csv` | PSO / DE parameter grid (16 configurations or checks) | final (sensitivity) |
| `time_matched_DE_PSO.json/.csv`, `time_trace_DE_PSO_RMMDE.csv` | seed-matched wall-clock comparison with PSO and DE (720 paired runs) | final |
| `ablation_A.json`, `ablation_A_runs.csv`, `ablation_A_trace.csv` | donor x local-search ablation (MCET) | final |
| `ablation_A2.json`, `ablation_A2_runs.csv`, `ablation_A2_trace.csv` | synergetic donor with fixed F and with F = Cr = 0.25 | final |
| `ablation_B.json`, `ablation_B_runs.csv`, `ablation_B_trace.csv` | p_ls sensitivity (MCET) | final |
| `multistart_C.json`, `multistart_C_runs.csv`, `multistart_C_trace.csv` | RM-MDE vs multistart exact local search (720 paired runs) | final |
| `stats_image_level.json`, `stats_image_level_master.csv`, `stats_image_level_perimage.csv` | image-level statistics (Wilcoxon, Hodges-Lehmann, rank-biserial, Holm, Friedman, bootstrap CIs) | final. Regenerating from this package reproduces `stats_image_level_master.csv` byte for byte; the regenerated JSON differs from the archived one only in path-dependent protocol metadata (the folder names quoted in its `protocol` text) and all numerical fields are identical, so the JSON is **not** claimed to be byte-identical |
| `runtime_A_L256.json`, `runtime_B_scaling.json` | DP vs RM-MDE runtime at L = 256 and scaling with the number of gray levels | final |

Trace files (`*_trace.csv`) store best-so-far values and population diversity per generation or per improvement; run files (`*_runs.csv`) store one row per run.
