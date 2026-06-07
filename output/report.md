# Analysis report

This report summarizes the System GMM analysis of political repression on GDP per capita.

## Files generated

- `clean_panel.csv` — processed panel used for estimation
- `system_gmm_coefficients.csv` — GMM coefficient table (main run)
- `main_results.csv` — summary table comparing TWFE, IV-FE, System GMM, DID
- `event_study.csv` and `event_study_plot.png` — event-study table and plot
- `gmm_wide_grid_clean_repression.csv` — grid search over GMM lag specs (one-step, collapse on/off)
- `coeff_comparison.png` — publication-ready comparison of TWFE / IV-FE / System GMM
- `gmm_lag_grid_plot.png` — simple lag-grid plot

## Key result (main run)

- System GMM (`gmm(ln_gdp_pc repression, 2:4)`) estimate for `repression`: **-0.01157** (SE 0.00666, p=0.082)

## How to reproduce

From the repository root:

```bash
pip install -r requirements.txt
python analysis_pipeline.py --config config.yaml
```

To run the alternative repression (`v2clkill`) pipeline:

```bash
python analysis_pipeline.py --config config_alt.yaml
```


## Notes

- System GMM was run via `pydynpd.regression.abond` and instruments were specified as in the config.
- Diagnostics were partially patched for compatibility; see `analysis_pipeline.py::patch_pydynpd_compat()`.


---

Generated on: 2026-06-07
