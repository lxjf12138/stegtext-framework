# Analysis scripts

These scripts turned the main-comparison run records into the tables and
figures of Section V. Their default paths follow the layout in which they ran:
run records under `reports/` and `runs/` of this repository, and derived data
and figures under `evidence/` and `manuscript/`. The run records are not
distributed.

| Order | Script | Input | Output |
|---|---|---|---|
| 1 | `verify_stage3.py` | sender, receiver, and score records of the main comparison and its two repeats | per-case records with recomputed payload and token counts (`CORRECTED_CASES.json.gz`) |
| 2 | `prepare_experiment_data.py` | per-case records | aggregate CSVs: recovery, operating points, perplexity ratios with prompt-cluster bootstrap intervals, synchronized share, threshold scan |
| 3 | `measure_ambiguity_frequency.py` | ordinary-sampling records and the two models | frequency of tokenization ambiguity per sampling step |
| 4 | `plot_experiments.py`, `write_experiment_tables.py` | aggregate CSVs | figures and LaTeX tables |

Steps 1, 2, and 4 run on the CPU; step 3 replays the recorded sequences on the
GPU.
