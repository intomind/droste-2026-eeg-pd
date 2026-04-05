#!/usr/bin/env python3
"""
experiment1_lifespan.py — Spectral analysis across the lifespan.

Reads results/features.csv and produces publication-quality analyses of
age-related EEG spectral patterns across the heterogeneous multi-study corpus.

Analyses:
1. Correlation heatmap (channel × band vs age) with Spearman r and Bonferroni correction
2. Posterior alpha power vs age (scatter + LOESS)
3. Peak alpha frequency vs age (scatter + LOESS)
4. Per-dataset consistency (forest plot)
5. Sex differences heatmap (Cohen's d with bootstrap CIs)
6. Hemispheric alpha asymmetry & BDI correlation

All analyses report effect sizes with 95% CIs.
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

# ── Constants ────────────────────────────────────────────────────────────────
CANONICAL_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
POSTERIOR_CHANNELS = ["P3", "Pz", "P4", "O1", "O2"]

# Plot defaults
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# ── Utility functions ────────────────────────────────────────────────────────

def spearman_with_ci(x, y, alpha=0.05):
    """Spearman r with Fisher z-transform 95% CI."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, (np.nan, np.nan)
    r, p = stats.spearmanr(x, y)
    # Fisher z-transform CI
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    return r, p, (ci_lo, ci_hi)


def cohens_d_bootstrap(group1, group2, n_boot=10000, alpha=0.05, rng=None):
    """Cohen's d with bootstrap 95% CI."""
    if rng is None:
        rng = np.random.default_rng(42)
    g1 = group1[np.isfinite(group1)]
    g2 = group2[np.isfinite(group2)]
    if len(g1) < 5 or len(g2) < 5:
        return np.nan, (np.nan, np.nan)

    # Pooled SD
    n1, n2 = len(g1), len(g2)
    pooled_sd = np.sqrt(
        ((n1 - 1) * g1.std(ddof=1)**2 + (n2 - 1) * g2.std(ddof=1)**2)
        / (n1 + n2 - 2)
    )
    if pooled_sd < 1e-12:
        return 0.0, (0.0, 0.0)
    d = (g1.mean() - g2.mean()) / pooled_sd

    # Bootstrap CI
    boot_ds = np.empty(n_boot)
    for i in range(n_boot):
        b1 = rng.choice(g1, size=n1, replace=True)
        b2 = rng.choice(g2, size=n2, replace=True)
        bp_sd = np.sqrt(
            ((n1 - 1) * b1.std(ddof=1)**2 + (n2 - 1) * b2.std(ddof=1)**2)
            / (n1 + n2 - 2)
        )
        if bp_sd < 1e-12:
            boot_ds[i] = 0.0
        else:
            boot_ds[i] = (b1.mean() - b2.mean()) / bp_sd

    ci_lo = np.percentile(boot_ds, 100 * alpha / 2)
    ci_hi = np.percentile(boot_ds, 100 * (1 - alpha / 2))
    return d, (ci_lo, ci_hi)


# ── Analysis 1: Correlation heatmap ──────────────────────────────────────────

def analysis_correlation_heatmap(df, stats_out):
    """Channel × band Spearman correlation with age."""
    print("\n=== Analysis 1: Age correlation heatmap ===")
    age = df["age_years"].values

    n_bands = len(BAND_NAMES)
    n_ch = len(CANONICAL_CHANNELS)
    r_matrix = np.full((n_bands, n_ch), np.nan)
    p_matrix = np.full((n_bands, n_ch), np.nan)
    ci_lo_matrix = np.full((n_bands, n_ch), np.nan)
    ci_hi_matrix = np.full((n_bands, n_ch), np.nan)
    n_matrix = np.full((n_bands, n_ch), 0, dtype=int)

    bonferroni = 0.05 / (n_bands * n_ch)
    corr_details = {}

    for b_idx, band in enumerate(BAND_NAMES):
        for ch_idx, ch in enumerate(CANONICAL_CHANNELS):
            col = f"{band}_{ch}"
            vals = df[col].values
            mask = np.isfinite(vals) & np.isfinite(age)
            n_valid = mask.sum()
            n_matrix[b_idx, ch_idx] = n_valid
            r, p, (ci_lo, ci_hi) = spearman_with_ci(vals, age)
            r_matrix[b_idx, ch_idx] = r
            p_matrix[b_idx, ch_idx] = p
            ci_lo_matrix[b_idx, ch_idx] = ci_lo
            ci_hi_matrix[b_idx, ch_idx] = ci_hi
            corr_details[col] = {
                "r": float(r) if np.isfinite(r) else None,
                "p": float(p) if np.isfinite(p) else None,
                "ci_95": [float(ci_lo), float(ci_hi)] if np.isfinite(ci_lo) else None,
                "n": int(n_valid),
                "significant_bonferroni": bool(p < bonferroni) if np.isfinite(p) else False,
            }

    stats_out["age_correlations"] = corr_details
    stats_out["bonferroni_threshold"] = bonferroni

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = np.nanmax(np.abs(r_matrix))
    im = ax.imshow(r_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate
    for b_idx in range(n_bands):
        for ch_idx in range(n_ch):
            r = r_matrix[b_idx, ch_idx]
            p = p_matrix[b_idx, ch_idx]
            if np.isfinite(r):
                star = "*" if p < bonferroni else ""
                color = "white" if abs(r) > vmax * 0.6 else "black"
                ax.text(ch_idx, b_idx, f"{r:.2f}{star}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold" if star else "normal")

    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(CANONICAL_CHANNELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_bands))
    ax.set_yticklabels([b.capitalize() for b in BAND_NAMES])
    ax.set_title(f"Spearman correlation: band power vs age (N = {len(df):,})\n"
                 f"* = significant after Bonferroni correction (p < {bonferroni:.1e})")
    plt.colorbar(im, ax=ax, label="Spearman r", shrink=0.8)

    fig.savefig(EXP1_DIR / "correlation_heatmap.png")
    plt.close(fig)
    print(f"  Saved correlation_heatmap.png")
    print(f"  Max |r|: {vmax:.3f}")
    n_sig = np.sum(p_matrix < bonferroni)
    print(f"  Significant (Bonferroni): {n_sig}/{n_bands * n_ch}")


# ── Analysis 2: Alpha power vs age ──────────────────────────────────────────

def analysis_alpha_power_vs_age(df, stats_out):
    """Posterior alpha power vs age with LOESS and per-dataset coloring."""
    print("\n=== Analysis 2: Posterior alpha power vs age ===")

    posterior_cols = [f"alpha_{ch}" for ch in POSTERIOR_CHANNELS]
    df_plot = df.dropna(subset=posterior_cols + ["age_years"]).copy()
    df_plot["posterior_alpha"] = df_plot[posterior_cols].mean(axis=1)

    age = df_plot["age_years"].values
    alpha = df_plot["posterior_alpha"].values

    r, p, ci = spearman_with_ci(age, alpha)
    stats_out["posterior_alpha_vs_age"] = {
        "r": float(r), "p": float(p), "ci_95": [float(ci[0]), float(ci[1])],
        "n": len(df_plot),
    }
    print(f"  N = {len(df_plot)}, Spearman r = {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], p = {p:.2e}")

    # LOESS
    sort_idx = np.argsort(age)
    loess_result = lowess(alpha[sort_idx], age[sort_idx], frac=0.3, return_sorted=True)

    # Top 10 datasets by subject count
    ds_counts = df_plot["dataset_id"].value_counts()
    top_datasets = ds_counts.head(10).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Color map for top datasets
    cmap = plt.cm.tab10
    colors = {}
    for i, ds in enumerate(top_datasets):
        colors[ds] = cmap(i)

    # Plot "other" datasets first
    other = df_plot[~df_plot["dataset_id"].isin(top_datasets)]
    if len(other) > 0:
        ax.scatter(other["age_years"], other["posterior_alpha"],
                   c="lightgray", s=8, alpha=0.4, label=f"Other ({len(other)})", rasterized=True)

    # Plot top datasets
    for ds in top_datasets:
        mask = df_plot["dataset_id"] == ds
        n_ds = mask.sum()
        ax.scatter(df_plot.loc[mask, "age_years"], df_plot.loc[mask, "posterior_alpha"],
                   c=[colors[ds]], s=12, alpha=0.5, label=f"{ds} (n={n_ds})", rasterized=True)

    # LOESS curve
    ax.plot(loess_result[:, 0], loess_result[:, 1], "k-", linewidth=2.5, label="LOESS")

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Relative alpha power (posterior)")
    ax.set_title(f"Posterior alpha power vs age\n"
                 f"r = {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], N = {len(df_plot):,}")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.8)
    ax.set_xlim(0, 95)

    fig.savefig(EXP1_DIR / "alpha_power_vs_age.png")
    plt.close(fig)
    print(f"  Saved alpha_power_vs_age.png")


# ── Analysis 3: Peak alpha frequency vs age ──────────────────────────────────

def analysis_peak_alpha_vs_age(df, stats_out):
    """Peak alpha frequency (posterior) vs age."""
    print("\n=== Analysis 3: Peak alpha frequency vs age ===")

    posterior_pa_cols = [f"peak_alpha_{ch}" for ch in POSTERIOR_CHANNELS]
    df_plot = df.dropna(subset=posterior_pa_cols + ["age_years"]).copy()
    df_plot["posterior_peak_alpha"] = df_plot[posterior_pa_cols].mean(axis=1)

    age = df_plot["age_years"].values
    paf = df_plot["posterior_peak_alpha"].values

    r, p, ci = spearman_with_ci(age, paf)
    stats_out["peak_alpha_freq_vs_age"] = {
        "r": float(r), "p": float(p), "ci_95": [float(ci[0]), float(ci[1])],
        "n": len(df_plot),
    }
    print(f"  N = {len(df_plot)}, Spearman r = {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], p = {p:.2e}")

    # LOESS
    sort_idx = np.argsort(age)
    loess_result = lowess(paf[sort_idx], age[sort_idx], frac=0.3, return_sorted=True)

    # Top datasets
    ds_counts = df_plot["dataset_id"].value_counts()
    top_datasets = ds_counts.head(10).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.tab10
    colors = {ds: cmap(i) for i, ds in enumerate(top_datasets)}

    other = df_plot[~df_plot["dataset_id"].isin(top_datasets)]
    if len(other) > 0:
        ax.scatter(other["age_years"], other["posterior_peak_alpha"],
                   c="lightgray", s=8, alpha=0.4, label=f"Other ({len(other)})", rasterized=True)

    for ds in top_datasets:
        mask = df_plot["dataset_id"] == ds
        n_ds = mask.sum()
        ax.scatter(df_plot.loc[mask, "age_years"], df_plot.loc[mask, "posterior_peak_alpha"],
                   c=[colors[ds]], s=12, alpha=0.5, label=f"{ds} (n={n_ds})", rasterized=True)

    ax.plot(loess_result[:, 0], loess_result[:, 1], "k-", linewidth=2.5, label="LOESS")

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Peak alpha frequency (Hz)")
    ax.set_title(f"Peak alpha frequency vs age (posterior channels)\n"
                 f"r = {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], N = {len(df_plot):,}")
    ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.8)
    ax.set_xlim(0, 95)
    ax.set_ylim(6, 14)

    fig.savefig(EXP1_DIR / "peak_alpha_freq_vs_age.png")
    plt.close(fig)
    print(f"  Saved peak_alpha_freq_vs_age.png")


# ── Analysis 4: Per-dataset consistency ──────────────────────────────────────

def analysis_dataset_consistency(df, stats_out):
    """Forest plot of per-dataset alpha-age correlations."""
    print("\n=== Analysis 4: Per-dataset consistency ===")

    posterior_cols = [f"alpha_{ch}" for ch in POSTERIOR_CHANNELS]
    df_sub = df.dropna(subset=posterior_cols + ["age_years"]).copy()
    df_sub["posterior_alpha"] = df_sub[posterior_cols].mean(axis=1)

    # Get datasets with 10+ subjects with age
    ds_counts = df_sub.groupby("dataset_id").size()
    large_ds = ds_counts[ds_counts >= 10].index.tolist()
    # Take top 10 by size
    large_ds = ds_counts[ds_counts >= 10].sort_values(ascending=False).head(10).index.tolist()

    # Global correlation
    global_r, global_p, global_ci = spearman_with_ci(
        df_sub["age_years"].values, df_sub["posterior_alpha"].values
    )

    results = []
    for ds in large_ds:
        ds_df = df_sub[df_sub["dataset_id"] == ds]
        age_vals = ds_df["age_years"].values
        alpha_vals = ds_df["posterior_alpha"].values
        r, p, ci = spearman_with_ci(age_vals, alpha_vals)
        results.append({
            "dataset_id": ds, "n": len(ds_df), "r": r, "p": p,
            "ci_lo": ci[0], "ci_hi": ci[1],
            "mean_age": float(age_vals.mean()),
        })

    results.sort(key=lambda x: x["mean_age"])

    stats_out["dataset_consistency"] = {
        "global_r": float(global_r),
        "global_p": float(global_p),
        "global_ci_95": [float(global_ci[0]), float(global_ci[1])],
        "per_dataset": results,
    }

    # Forest plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.5 + 2)))

    y_pos = range(len(results))
    for i, res in enumerate(results):
        color = "tab:blue" if res["ci_hi"] < 0 or res["ci_lo"] > 0 else "tab:gray"
        ax.errorbarx = ax.errorbar(
            res["r"], i, xerr=[[res["r"] - res["ci_lo"]], [res["ci_hi"] - res["r"]]],
            fmt="o", color=color, capsize=3, markersize=6,
        )

    ax.axvline(global_r, color="red", linestyle="--", linewidth=1.5,
               label=f"Global r = {global_r:.3f}")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    labels = [f"{r['dataset_id']} (n={r['n']}, age={r['mean_age']:.0f})" for r in results]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Spearman r (posterior alpha vs age)")
    ax.set_title("Per-dataset correlation: posterior alpha power vs age\n"
                 f"Global r = {global_r:.3f} [{global_ci[0]:.3f}, {global_ci[1]:.3f}]")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    fig.savefig(EXP1_DIR / "dataset_consistency.png")
    plt.close(fig)
    print(f"  Saved dataset_consistency.png ({len(results)} datasets)")


# ── Analysis 5: Sex differences heatmap ──────────────────────────────────────

def analysis_sex_differences(df, stats_out):
    """Cohen's d (male - female) for each band×channel with bootstrap CIs."""
    print("\n=== Analysis 5: Sex differences heatmap ===")

    df_sex = df[df["sex"].isin(["male", "female"])].copy()
    n_male = (df_sex["sex"] == "male").sum()
    n_female = (df_sex["sex"] == "female").sum()
    print(f"  N = {len(df_sex)} ({n_male} male, {n_female} female)")

    if len(df_sex) < 20:
        print("  Too few subjects with sex labels, skipping")
        stats_out["sex_differences"] = {"skipped": True, "reason": "too few subjects"}
        return

    n_bands = len(BAND_NAMES)
    n_ch = len(CANONICAL_CHANNELS)
    d_matrix = np.full((n_bands, n_ch), np.nan)
    ci_lo_matrix = np.full((n_bands, n_ch), np.nan)
    ci_hi_matrix = np.full((n_bands, n_ch), np.nan)

    rng = np.random.default_rng(42)
    sex_details = {}

    for b_idx, band in enumerate(BAND_NAMES):
        for ch_idx, ch in enumerate(CANONICAL_CHANNELS):
            col = f"{band}_{ch}"
            male_vals = df_sex.loc[df_sex["sex"] == "male", col].dropna().values
            female_vals = df_sex.loc[df_sex["sex"] == "female", col].dropna().values
            d, (ci_lo, ci_hi) = cohens_d_bootstrap(male_vals, female_vals, rng=rng)
            d_matrix[b_idx, ch_idx] = d
            ci_lo_matrix[b_idx, ch_idx] = ci_lo
            ci_hi_matrix[b_idx, ch_idx] = ci_hi
            sex_details[col] = {
                "d": float(d) if np.isfinite(d) else None,
                "ci_95": [float(ci_lo), float(ci_hi)] if np.isfinite(ci_lo) else None,
                "n_male": int(len(male_vals)),
                "n_female": int(len(female_vals)),
                "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0) if np.isfinite(ci_lo) else False,
            }

    stats_out["sex_differences"] = {
        "n_male": int(n_male), "n_female": int(n_female),
        "per_feature": sex_details,
    }

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = np.nanmax(np.abs(d_matrix))
    vmax = max(vmax, 0.1)  # ensure some range
    im = ax.imshow(d_matrix, cmap="PiYG", vmin=-vmax, vmax=vmax, aspect="auto")

    for b_idx in range(n_bands):
        for ch_idx in range(n_ch):
            d = d_matrix[b_idx, ch_idx]
            ci_lo = ci_lo_matrix[b_idx, ch_idx]
            ci_hi = ci_hi_matrix[b_idx, ch_idx]
            if np.isfinite(d):
                excludes_zero = (ci_lo > 0 or ci_hi < 0)
                star = "*" if excludes_zero else ""
                color = "white" if abs(d) > vmax * 0.6 else "black"
                ax.text(ch_idx, b_idx, f"{d:.2f}{star}", ha="center", va="center",
                        fontsize=8, color=color,
                        fontweight="bold" if star else "normal")

    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(CANONICAL_CHANNELS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_bands))
    ax.set_yticklabels([b.capitalize() for b in BAND_NAMES])
    ax.set_title(f"Sex differences: Cohen's d (male - female)\n"
                 f"N = {n_male} male, {n_female} female  |  * = 95% CI excludes zero")
    plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)

    fig.savefig(EXP1_DIR / "sex_differences_heatmap.png")
    plt.close(fig)
    print(f"  Saved sex_differences_heatmap.png")


# ── Analysis 6: Alpha asymmetry & BDI ────────────────────────────────────────

def analysis_alpha_asymmetry(df, stats_out):
    """Hemispheric alpha asymmetry and correlation with BDI."""
    print("\n=== Analysis 6: Alpha asymmetry & BDI ===")

    # Need F3, F4, P3, P4 to not be NaN
    needed = ["alpha_F3", "alpha_F4", "alpha_P3", "alpha_P4"]
    df_asym = df.dropna(subset=needed).copy()

    # Use log of relative band power — add small epsilon to avoid log(0)
    eps = 1e-10
    df_asym["frontal_asymmetry"] = (
        np.log(df_asym["alpha_F4"].clip(lower=eps))
        - np.log(df_asym["alpha_F3"].clip(lower=eps))
    )
    df_asym["parietal_asymmetry"] = (
        np.log(df_asym["alpha_P4"].clip(lower=eps))
        - np.log(df_asym["alpha_P3"].clip(lower=eps))
    )

    asym_stats = {
        "n_total": len(df_asym),
        "frontal_asymmetry_mean": float(df_asym["frontal_asymmetry"].mean()),
        "frontal_asymmetry_std": float(df_asym["frontal_asymmetry"].std()),
        "parietal_asymmetry_mean": float(df_asym["parietal_asymmetry"].mean()),
        "parietal_asymmetry_std": float(df_asym["parietal_asymmetry"].std()),
    }

    # BDI correlation
    df_bdi = df_asym.dropna(subset=["score_bdi"])
    asym_stats["n_with_bdi"] = len(df_bdi)
    print(f"  N with asymmetry data: {len(df_asym)}")
    print(f"  N with BDI scores: {len(df_bdi)}")

    if len(df_bdi) >= 20:
        bdi = df_bdi["score_bdi"].values
        frontal = df_bdi["frontal_asymmetry"].values
        parietal = df_bdi["parietal_asymmetry"].values

        r_front, p_front, ci_front = spearman_with_ci(frontal, bdi)
        r_pari, p_pari, ci_pari = spearman_with_ci(parietal, bdi)

        asym_stats["frontal_bdi"] = {
            "r": float(r_front), "p": float(p_front),
            "ci_95": [float(ci_front[0]), float(ci_front[1])],
        }
        asym_stats["parietal_bdi"] = {
            "r": float(r_pari), "p": float(p_pari),
            "ci_95": [float(ci_pari[0]), float(ci_pari[1])],
        }

        print(f"  Frontal asymmetry vs BDI: r = {r_front:.3f} "
              f"[{ci_front[0]:.3f}, {ci_front[1]:.3f}], p = {p_front:.2e}")
        print(f"  Parietal asymmetry vs BDI: r = {r_pari:.3f} "
              f"[{ci_pari[0]:.3f}, {ci_pari[1]:.3f}], p = {p_pari:.2e}")

        # Plot if either is noteworthy (p < 0.05 or effect size > 0.1)
        if p_front < 0.05 or abs(r_front) > 0.1:
            sort_idx = np.argsort(bdi)
            loess_result = lowess(frontal[sort_idx], bdi[sort_idx], frac=0.5, return_sorted=True)

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(bdi, frontal, c="steelblue", s=15, alpha=0.5, rasterized=True)
            ax.plot(loess_result[:, 0], loess_result[:, 1], "k-", linewidth=2.5, label="LOESS")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("BDI Score")
            ax.set_ylabel("Frontal alpha asymmetry [ln(F4) - ln(F3)]")
            ax.set_title(f"Frontal alpha asymmetry vs depression (BDI)\n"
                         f"r = {r_front:.3f} [{ci_front[0]:.3f}, {ci_front[1]:.3f}], "
                         f"N = {len(df_bdi)}")
            ax.legend()

            fig.savefig(EXP1_DIR / "alpha_asymmetry_bdi.png")
            plt.close(fig)
            print(f"  Saved alpha_asymmetry_bdi.png")
        else:
            print(f"  Frontal asymmetry-BDI not plotted (p={p_front:.3f}, |r|={abs(r_front):.3f})")
    else:
        print("  Too few subjects with BDI to test asymmetry correlation")
        asym_stats["bdi_skipped"] = True

    stats_out["alpha_asymmetry"] = asym_stats


# ── Pre-analysis report ──────────────────────────────────────────────────────

def pre_analysis_report(df):
    """Print summary statistics before running analyses."""
    print("=" * 70)
    print("PRE-ANALYSIS REPORT")
    print("=" * 70)

    n_total = len(df)
    n_with_age = df["age_years"].notna().sum()
    n_datasets = df["dataset_id"].nunique()
    n_datasets_with_age = df[df["age_years"].notna()]["dataset_id"].nunique()

    print(f"Total subjects in features.csv: {n_total:,}")
    print(f"Subjects with numeric age: {n_with_age:,}")
    print(f"Datasets total: {n_datasets}")
    print(f"Datasets contributing to age analysis: {n_datasets_with_age}")

    # Age distribution
    ages = df["age_years"].dropna()
    print(f"\nAge distribution (N={len(ages):,}):")
    print(f"  Min: {ages.min():.1f}, Max: {ages.max():.1f}")
    print(f"  Mean: {ages.mean():.1f}, Median: {ages.median():.1f}")
    print(f"  Std: {ages.std():.1f}")
    bins = [0, 5, 10, 18, 30, 50, 65, 100]
    labels = ["0-5", "5-10", "10-18", "18-30", "30-50", "50-65", "65+"]
    age_groups = pd.cut(ages, bins=bins, labels=labels, right=False)
    print("  Distribution:")
    for label in labels:
        n = (age_groups == label).sum()
        print(f"    {label:8s}: {n:5d} ({100*n/len(ages):.1f}%)")

    # Sex distribution
    print(f"\nSex distribution:")
    for s in df["sex"].value_counts().items():
        print(f"  {s[0]:15s}: {s[1]:5d}")

    # Group distribution
    print(f"\nGroup labels (top 15):")
    groups = df[df["group_label"].notna()].groupby(
        ["group_label", "group_role"]
    ).size().sort_values(ascending=False).head(15)
    for (label, role), n in groups.items():
        print(f"  {label:35s} ({role:10s}): {n:5d}")

    # BDI scores
    n_bdi = df["score_bdi"].notna().sum()
    print(f"\nSubjects with BDI scores: {n_bdi}")

    print("=" * 70)

    return {
        "n_total": int(n_total),
        "n_with_age": int(n_with_age),
        "n_datasets": int(n_datasets),
        "n_datasets_with_age": int(n_datasets_with_age),
        "age_min": float(ages.min()),
        "age_max": float(ages.max()),
        "age_mean": float(ages.mean()),
        "age_median": float(ages.median()),
        "n_bdi": int(n_bdi),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spectral analysis across the lifespan (Experiment 1).",
    )
    parser.add_argument(
        "features_csv", type=Path,
        help="Path to features.csv (subject-level spectral features).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory for figures and stats JSON. "
             "Defaults to a sibling 'experiment1/' directory next to features_csv.",
    )
    args = parser.parse_args()

    features_csv = args.features_csv.resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = features_csv.parent / "experiment1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make output_dir available to analysis functions via module-level name
    global EXP1_DIR
    EXP1_DIR = output_dir

    print("Loading features.csv...")
    df = pd.read_csv(features_csv)
    print(f"  Loaded {len(df)} subjects")

    stats_out = {}
    stats_out["summary"] = pre_analysis_report(df)

    # Filter to subjects with age for most analyses
    df_age = df[df["age_years"].notna()].copy()
    print(f"\nProceeding with {len(df_age):,} subjects that have numeric age\n")

    analysis_correlation_heatmap(df_age, stats_out)
    analysis_alpha_power_vs_age(df_age, stats_out)
    analysis_peak_alpha_vs_age(df_age, stats_out)
    analysis_dataset_consistency(df_age, stats_out)
    analysis_sex_differences(df, stats_out)  # uses all subjects with sex labels
    analysis_alpha_asymmetry(df, stats_out)  # uses all subjects with BDI

    # Save stats
    with open(output_dir / "experiment1_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2, default=str)
    print(f"\nSaved experiment1_stats.json")
    print("\nExperiment 1 complete!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
