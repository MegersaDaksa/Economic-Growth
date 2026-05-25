# Causal Impact of Political Repression on Income Per Capita

This project provides an end-to-end empirical pipeline for unbalanced panel data.

## Empirical Design

1. Two-way fixed effects (country and year FE)
2. IV fixed effects (2SLS with country and year FE)
3. Dynamic panel System GMM
4. DID robustness (staggered first repression-shock adoption)

Dependent variable: log income per capita (`ln_gdp_pc`).

## Data

Input file currently configured in `config.yaml`:

- `combined_GDP_growth_1970_2024_with_VDem.xlsx`

## Run

```powershell
.\.venv\Scripts\python.exe analysis_pipeline.py --config config.yaml
```

## Outputs

All outputs are written to `output/`:

- `clean_panel.csv`: cleaned estimation sample
- `main_results.csv`: compact comparison table across models
- `twfe_full.txt`: full TWFE summary
- `iv_fe_full.txt`: full IV-FE summary
- `did_full.txt`: full DID summary
- `event_study.csv`: event-time coefficients for DID diagnostics
- `event_study_plot.png`: event-study figure
- `system_gmm_coefficients.csv`: System GMM coefficient table
- `run_log.txt`: execution log and warnings

## Notes on System GMM

`pydynpd` currently has known compatibility issues with modern NumPy/Python combinations for some post-estimation tests.
The pipeline applies a runtime compatibility patch so estimation still runs and coefficients are exported.
For final journal submission, replicate dynamic panel estimates in Stata (`xtabond2`) or R (`pgmm`) as a cross-check.
