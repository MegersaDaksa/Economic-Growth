#!/usr/bin/env python3
"""
Run (system) GMM on panel data, optimize instrument lags, and save results + plots.

Usage examples:
  python analysis_system_gmm.py \
    --input combined_GDP_growth_1970_2024_with_VDem.xlsx \
    --entity country --time year \
    --depvar gdp_per_capita --main poltical_repression \
    --controls control1 control2

The script will try `pydynpd` (system GMM) first, then fall back to
`linearmodels` Arellano-Bond (difference GMM) if available. It saves
results and plots to the `output/` folder.
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    plt.style.use('seaborn-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn')
    except Exception:
        pass


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df


def fit_with_pydynpd(df, depvar, exog, entity, time, lag_range):
    try:
        # Attempt to import pydynpd (recommended for system GMM)
        import pydynpd as pdyn
    except Exception:
        return None, 'pydynpd not available'

    try:
        # pydynpd API may differ across versions; try a common pattern
        # Build and fit a System GMM model using pydynpd if available.
        model = pdyn.SystemGMM(df, entity=entity, time=time,
                                depvar=depvar, exog=exog,
                                lag_min=lag_range[0], lag_max=lag_range[1])
        res = model.fit()
        return res, None
    except Exception as e:
        return None, str(e)


def fit_with_linearmodels(df, formula, entity, time, lags=(2, 4)):
    try:
        from linearmodels.panel import ArellanoBond
    except Exception as e:
        return None, f'linearmodels or ArellanoBond unavailable: {e}'

    try:
        # Attempt to use the formula interface if present
        m = ArellanoBond.from_formula(formula, data=df)
        res = m.fit()
        return res, None
    except Exception as e:
        return None, str(e)


def save_model_summary(res, out_dir, name_prefix="gmm"):
    txt = None
    try:
        txt = str(res.summary)
    except Exception:
        try:
            txt = str(res)
        except Exception:
            txt = "<no summary available>"

    fname = os.path.join(out_dir, f"{name_prefix}_summary.txt")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(txt)
    return fname


def coef_dataframe(res):
    try:
        params = res.params
        se = res.std_errors
        tstats = res.tstats
        pvals = res.pvalues
        df = pd.DataFrame({'coef': params, 'se': se, 't': tstats, 'p': pvals})
        return df
    except Exception:
        # best-effort fallback
        try:
            s = res.summary
            return pd.DataFrame({'summary': [str(s)]})
        except Exception:
            return pd.DataFrame()


def plot_results(df, depvar, fitted, out_dir):
    ensure_output_dir(out_dir)
    # Actual vs fitted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=fitted, y=df[depvar])
    mn = min(df[depvar].min(), np.nanmin(fitted))
    mx = max(df[depvar].max(), np.nanmax(fitted))
    plt.plot([mn, mx], [mn, mx], color='k', linestyle='--')
    plt.xlabel('Fitted')
    plt.ylabel('Actual')
    plt.title('Actual vs Fitted')
    plt.tight_layout()
    p1 = os.path.join(out_dir, 'actual_vs_fitted.png')
    plt.savefig(p1)
    plt.close()

    # Residuals
    resid = df[depvar] - fitted
    plt.figure(figsize=(8, 4))
    sns.histplot(resid.dropna(), kde=True)
    plt.title('Residuals distribution')
    plt.tight_layout()
    p2 = os.path.join(out_dir, 'residuals_dist.png')
    plt.savefig(p2)
    plt.close()

    return [p1, p2]


def main(argv=None):
    parser = argparse.ArgumentParser(description='System GMM pipeline')
    parser.add_argument('--input', required=True, help='Excel filename in workspace')
    parser.add_argument('--sheet', default=0)
    parser.add_argument('--entity', default='country')
    parser.add_argument('--time', default='year')
    parser.add_argument('--depvar', default='gdp_per_capita')
    parser.add_argument('--main', default='poltical_repression')
    parser.add_argument('--controls', nargs='*', default=[])
    parser.add_argument('--outdir', default='output')
    parser.add_argument('--opt-lag-min', type=int, default=2)
    parser.add_argument('--opt-lag-max', type=int, default=4)
    args = parser.parse_args(argv)

    input_path = os.path.join(os.path.dirname(__file__), args.input)
    if not os.path.exists(input_path):
        print(f'Input file not found: {input_path}', file=sys.stderr)
        sys.exit(1)

    df = load_data(input_path, sheet_name=args.sheet)
    print('Loaded data with shape', df.shape)

    # Basic cleaning: drop rows without entity/time/depvar
    needed_cols = [args.entity, args.time, args.depvar, args.main] + args.controls
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print('Missing columns:', missing, file=sys.stderr)
        print('Available columns:', list(df.columns))
        sys.exit(1)

    df = df.dropna(subset=[args.entity, args.time, args.depvar])
    # Sort and set index for panel methods that expect it
    df = df.sort_values([args.entity, args.time]).copy()

    # We'll keep a working copy for fitting
    fit_df = df.copy()

    # Build formula for linearmodels if needed
    rhs = ' + '.join([args.main] + args.controls)
    formula = f"{args.depvar} ~ 1 + {rhs}"

    ensure_output_dir(args.outdir)

    best_res = None
    best_info = None

    # Grid search over lag settings
    for lag_min in range(args.opt_lag_min, args.opt_lag_max + 1):
        lag_range = (lag_min, args.opt_lag_max)
        print('Trying pydynpd SystemGMM with lag_range=', lag_range)
        res, err = fit_with_pydynpd(fit_df, args.depvar, [args.main] + args.controls,
                                    args.entity, args.time, lag_range)
        if res is not None:
            print('pydynpd fit succeeded for lag_range', lag_range)
            best_res = res
            best_info = ('pydynpd', lag_range)
            break
        else:
            print('pydynpd error:', err)

    if best_res is None:
        # Fall back to Arellano-Bond (difference GMM) using linearmodels
        print('Falling back to Arellano-Bond (difference GMM) via linearmodels')
        res, err = fit_with_linearmodels(fit_df, formula, args.entity, args.time,
                                         lags=(args.opt_lag_min, args.opt_lag_max))
        if res is None:
            print('All GMM estimators failed:', err, file=sys.stderr)
            print('You may need to install pydynpd or update column names.', file=sys.stderr)
            # Save a simple OLS as fallback
            import statsmodels.formula.api as smf
            ols = smf.ols(formula=formula, data=fit_df).fit()
            ols_fname = os.path.join(args.outdir, 'ols_summary.txt')
            with open(ols_fname, 'w', encoding='utf-8') as f:
                f.write(str(ols.summary()))
            print('Saved OLS fallback summary to', ols_fname)
            # plot actual vs fitted
            try:
                fitted = ols.fittedvalues
                plot_results(fit_df, args.depvar, fitted, args.outdir)
            except Exception:
                pass
            sys.exit(1)
        else:
            best_res = res
            best_info = ('linearmodels_arellano', (args.opt_lag_min, args.opt_lag_max))

    # Save results
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    prefix = f'gmm_{best_info[0]}_{timestamp}'
    summary_path = save_model_summary(best_res, args.outdir, name_prefix=prefix)
    print('Saved model summary to', summary_path)

    # Coefficients
    coef_df = coef_dataframe(best_res)
    coef_csv = os.path.join(args.outdir, f'{prefix}_coefficients.csv')
    coef_df.to_csv(coef_csv)
    print('Saved coefficients to', coef_csv)

    # Try to extract fitted values for plots; many GMM implementations don't provide them directly
    fitted = None
    try:
        fitted = best_res.predict()
    except Exception:
        try:
            fitted = best_res.fitted_values
        except Exception:
            # As last resort, compute linear predictor with params and data
            try:
                params = best_res.params
                X = pd.concat([pd.Series(1, index=fit_df.index, name='const'), fit_df[[args.main] + args.controls]], axis=1)
                fitted = X.dot(params.loc[X.columns])
            except Exception:
                fitted = pd.Series(np.nan, index=fit_df.index)

    # Attach fitted to the original df by index if possible
    try:
        fitted = pd.Series(fitted, index=fit_df.index)
    except Exception:
        pass

    plots = []
    try:
        plots = plot_results(fit_df, args.depvar, fitted, args.outdir)
    except Exception as e:
        print('Plotting error:', e)

    print('Plots saved:', plots)
    print('All done. Results and plots are in the', args.outdir, 'folder.')


if __name__ == '__main__':
    main()
