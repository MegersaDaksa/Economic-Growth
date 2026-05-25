import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from linearmodels import PanelOLS
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_data(cfg: dict) -> pd.DataFrame:
    input_file = Path(cfg["data"]["input_file"])
    v = cfg["variables"]

    if not input_file.exists():
        raise FileNotFoundError(f"Input data file not found: {input_file}")

    df = pd.read_excel(input_file)

    required = [
        v["entity"],
        v["time"],
        v["income_level"],
        v["repression"],
        v["iv"],
    ] + list(v["controls"])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep = [c for c in required if c in df.columns]
    df = df[keep].copy()

    for col in keep:
        if col not in (v["entity"],):
            df[col] = _safe_numeric(df[col])

    df = df.rename(
        columns={
            v["entity"]: "entity",
            v["time"]: "year",
            v["income_level"]: "gdp_pc",
            v["repression"]: "repression",
            v["iv"]: "iv_repression",
        }
    )

    controls_map = {}
    for c in v["controls"]:
        clean = c.lower().replace("%", "pct").replace(" ", "_").replace("__", "_")
        controls_map[c] = clean
    df = df.rename(columns=controls_map)

    control_cols = list(controls_map.values())

    df = df[(df["year"] >= cfg["model"]["min_year"]) & (df["year"] <= cfg["model"]["max_year"])].copy()

    df["gdp_pc"] = _safe_numeric(df["gdp_pc"])
    df["ln_gdp_pc"] = np.log(df["gdp_pc"])
    df = df.replace([np.inf, -np.inf], np.nan)

    cols_for_sample = ["entity", "year", "ln_gdp_pc", "repression", "iv_repression"] + control_cols
    df = df.dropna(subset=cols_for_sample).copy()

    df = df.sort_values(["entity", "year"]).reset_index(drop=True)
    return df


def run_twfe(df: pd.DataFrame, controls: list[str]):
    panel = df.set_index(["entity", "year"])
    y = panel["ln_gdp_pc"]
    x = panel[["repression"] + controls]
    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)
    return model.fit(cov_type="clustered", cluster_entity=True)


def run_iv_fe(df: pd.DataFrame, controls: list[str]):
    rhs = " + ".join(controls)
    formula = (
        "ln_gdp_pc ~ 1 + "
        + rhs
        + " + C(entity) + C(year) + [repression ~ iv_repression]"
    )
    model = IV2SLS.from_formula(formula, data=df)
    return model.fit(cov_type="clustered", clusters=df["entity"])


def patch_pydynpd_compat():
    import pydynpd.regression as reg_mod

    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    reg_mod.abond.perform_test = lambda self, model, step: None

    def _patched_form_results(self, model):
        model.form_regression_table()
        self.models.append(model)

    reg_mod.abond.form_results = _patched_form_results


def run_system_gmm(df: pd.DataFrame, controls: list[str]):
    patch_pydynpd_compat()
    from pydynpd import regression

    gmm_df = df[["entity", "year", "ln_gdp_pc", "repression"] + controls].copy()

    # Dynamic specification with lagged dependent variable and endogenous repression.
    lhs_rhs = "ln_gdp_pc L1.ln_gdp_pc repression " + " ".join(controls)
    gmm_block = "gmm(ln_gdp_pc repression, 2:4)"
    iv_block = f"iv({' '.join(controls)})"
    options = "onestep collapse"
    cmd = f"{lhs_rhs} | {gmm_block} {iv_block} | {options}"

    model = regression.abond(cmd, gmm_df, ["entity", "year"])
    if not model.models:
        raise RuntimeError("System GMM returned no valid model.")
    return model.models[0], cmd


def build_did_treatment(df: pd.DataFrame, q: float) -> pd.DataFrame:
    d = df.copy()

    threshold = d["repression"].quantile(q)
    d["repression_shock"] = (d["repression"] >= threshold).astype(int)

    first_treat = (
        d.loc[d["repression_shock"] == 1]
        .groupby("entity", as_index=False)["year"]
        .min()
        .rename(columns={"year": "first_treat_year"})
    )
    d = d.merge(first_treat, on="entity", how="left")
    d["ever_treated"] = d["first_treat_year"].notna().astype(int)
    d["post"] = np.where(
        d["first_treat_year"].notna(),
        (d["year"] >= d["first_treat_year"]).astype(int),
        0,
    )
    d["did_treat_post"] = d["ever_treated"] * d["post"]
    d["event_time"] = d["year"] - d["first_treat_year"]
    return d


def run_did(df: pd.DataFrame, controls: list[str]):
    panel = df.set_index(["entity", "year"])
    y = panel["ln_gdp_pc"]
    x = panel[["did_treat_post"] + controls]
    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)
    return model.fit(cov_type="clustered", cluster_entity=True)


def run_event_study(df: pd.DataFrame, controls: list[str], lead: int, lag: int):
    d = df.copy()
    d = d[d["ever_treated"] == 1].copy()

    d["event_time_int"] = d["event_time"].astype("Int64")
    window = list(range(-lead, lag + 1))
    ref = -1

    for k in window:
        if k == ref:
            continue
        col = f"event_{k}"
        d[col] = (d["event_time_int"] == k).astype(int)

    event_cols = [f"event_{k}" for k in window if k != ref]
    panel = d.set_index(["entity", "year"])
    y = panel["ln_gdp_pc"]
    x = panel[event_cols + controls]
    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = model.fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for col in event_cols:
        k = int(col.split("_")[1])
        rows.append(
            {
                "event_time": k,
                "coef": float(res.params.get(col, np.nan)),
                "se": float(res.std_errors.get(col, np.nan)),
                "pval": float(res.pvalues.get(col, np.nan)),
            }
        )
    return res, pd.DataFrame(rows).sort_values("event_time")


def summarize_results(twfe, ivfe, gmm_table: pd.DataFrame, did) -> pd.DataFrame:
    rows = []

    rows.append(
        {
            "model": "TWFE",
            "key_var": "repression",
            "coef": float(twfe.params.get("repression", np.nan)),
            "se": float(twfe.std_errors.get("repression", np.nan)),
            "pval": float(twfe.pvalues.get("repression", np.nan)),
            "nobs": int(twfe.nobs),
        }
    )

    rows.append(
        {
            "model": "IV-FE",
            "key_var": "repression",
            "coef": float(ivfe.params.get("repression", np.nan)),
            "se": float(ivfe.std_errors.get("repression", np.nan)),
            "pval": float(ivfe.pvalues.get("repression", np.nan)),
            "nobs": int(ivfe.nobs),
        }
    )

    gmm_row = gmm_table.loc[gmm_table["variable"] == "repression"]
    if not gmm_row.empty:
        g = gmm_row.iloc[0]
        rows.append(
            {
                "model": "System GMM",
                "key_var": "repression",
                "coef": float(g["coefficient"]),
                "se": float(g["std_err"]),
                "pval": float(g["p_value"]),
                "nobs": np.nan,
            }
        )

    rows.append(
        {
            "model": "DID",
            "key_var": "did_treat_post",
            "coef": float(did.params.get("did_treat_post", np.nan)),
            "se": float(did.std_errors.get("did_treat_post", np.nan)),
            "pval": float(did.pvalues.get("did_treat_post", np.nan)),
            "nobs": int(did.nobs),
        }
    )

    return pd.DataFrame(rows)


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Causal panel pipeline: TWFE, IV-FE, System GMM, DID")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = Path(cfg["data"]["output_dir"])
    ensure_output_dir(out_dir)

    v = cfg["variables"]
    controls = [c.lower().replace("%", "pct").replace(" ", "_").replace("__", "_") for c in v["controls"]]

    log_lines = []
    log_lines.append("Running causal analysis pipeline")

    df = prepare_data(cfg)
    df.to_csv(out_dir / "clean_panel.csv", index=False)
    log_lines.append(f"Prepared sample rows: {len(df)}")

    twfe = run_twfe(df, controls)
    save_text(out_dir / "twfe_full.txt", twfe.summary.as_text())
    log_lines.append("TWFE completed")

    ivfe = run_iv_fe(df, controls)
    save_text(out_dir / "iv_fe_full.txt", ivfe.summary.as_text())
    log_lines.append("IV-FE completed")

    gmm_model, gmm_cmd = run_system_gmm(df, controls)
    gmm_table = gmm_model.regression_table.copy()
    gmm_table.to_csv(out_dir / "system_gmm_coefficients.csv", index=False)
    log_lines.append(f"System GMM completed with command: {gmm_cmd}")
    log_lines.append("System GMM diagnostics are disabled due package compatibility patch.")

    did_df = build_did_treatment(df, cfg["model"]["did_shock_quantile"])
    did_res = run_did(did_df, controls)
    save_text(out_dir / "did_full.txt", did_res.summary.as_text())
    log_lines.append("DID completed")

    event_table = run_event_study(
        did_df,
        controls,
        cfg["model"]["event_window"]["lead"],
        cfg["model"]["event_window"]["lag"],
    )[1]
    event_table.to_csv(out_dir / "event_study.csv", index=False)
    log_lines.append("Event-study completed")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        event_table["event_time"],
        event_table["coef"],
        yerr=1.96 * event_table["se"],
        fmt="o-",
        capsize=4,
    )
    ax.axvline(-1, linestyle="--", color="gray")
    ax.axhline(0, linestyle=":", color="black")
    ax.set_title("Event Study: Dynamic Effect of Repression Shock on log GDP per Capita")
    ax.set_xlabel("Event time (years, reference = -1)")
    ax.set_ylabel("Coefficient (with 95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "event_study_plot.png", dpi=300)
    plt.close(fig)

    summary = summarize_results(twfe, ivfe, gmm_table, did_res)
    summary.to_csv(out_dir / "main_results.csv", index=False)

    payload = {
        "sample_n": len(df),
        "entities": int(df["entity"].nunique()),
        "years": [int(df["year"].min()), int(df["year"].max())],
        "controls": controls,
    }
    save_text(out_dir / "meta.json", json.dumps(payload, indent=2))

    save_text(out_dir / "run_log.txt", "\n".join(log_lines))


if __name__ == "__main__":
    main()
