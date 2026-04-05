#!/usr/bin/env python3
"""
experiment2_classification.py — Cross-dataset diagnosis classification.

Tests whether simple spectral features can separate clinical groups across
independent studies using leave-one-dataset-out cross-validation.

Core analyses:
1. Leave-one-dataset-out CV (Parkinson's vs neurotypical)
2. Stratified 5-fold CV (upper bound)
3. Age confound analysis (with/without age as feature)
4. Dataset confound analysis (dataset identity classifier + residualized features)
5. Feature importance (logistic regression coefficients, random forest importances)

Sensitivity analyses (run with --sensitivity):
6. Frontal channel removal (drop Fp1, Fp2, Fz features)
7. Gamma-band removal (drop all gamma features)
8. MoCA cognitive covariate analysis (4 datasets with MoCA scores)
9. Label permutation test (100 within-dataset permutations)
"""

import argparse
import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Constants ───────────────────────────��──────────────────────────────��─────
CANONICAL_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

BAND_POWER_COLS = [f"{b}_{ch}" for b in BAND_NAMES for ch in CANONICAL_CHANNELS]
PEAK_ALPHA_COLS = [f"peak_alpha_{ch}" for ch in CANONICAL_CHANNELS]
FEATURE_COLS = BAND_POWER_COLS + PEAK_ALPHA_COLS  # 114 features

PARKINSONS_DATASETS = [
    "ds003490", "ds003506", "ds003509",
    "ds004574", "ds004579", "ds004580", "ds004584",
]

RANDOM_STATE = 42

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


# ── Helper functions ─────────────────────────────────────────────────────────

def make_pipeline(model):
    """Standard pipeline: impute → scale → classify."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", model),
    ])


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_parkinsons_data(df):
    """Filter to Parkinson's datasets, return features and labels."""
    # Only subjects in the 7 Parkinson's datasets with patient/control roles
    mask = (
        df["dataset_id"].isin(PARKINSONS_DATASETS) &
        df["group_role"].isin(["patient", "control"])
    )
    df_pk = df[mask].copy()

    # Binary label: 1 = parkinsons, 0 = control
    df_pk["label"] = (df_pk["group_role"] == "patient").astype(int)

    print(f"Parkinson's classification data:")
    print(f"  Total: {len(df_pk)} subjects")
    for ds in PARKINSONS_DATASETS:
        ds_data = df_pk[df_pk["dataset_id"] == ds]
        n_pat = (ds_data["label"] == 1).sum()
        n_ctrl = (ds_data["label"] == 0).sum()
        print(f"  {ds}: {n_pat} patients, {n_ctrl} controls")
    print(f"  Overall: {(df_pk['label']==1).sum()} patients, "
          f"{(df_pk['label']==0).sum()} controls")

    return df_pk


# ── LODO CV ────────────────────────────────────────────────────��─────────────

def run_lodo_cv(df_pk, feature_cols, label="lodo"):
    """Leave-one-dataset-out cross-validation."""
    print(f"\n--- LODO CV ({label}, {len(feature_cols)} features) ---")

    X = df_pk[feature_cols].values
    y = df_pk["label"].values
    datasets = df_pk["dataset_id"].values

    models = {
        "logistic_regression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = {}
    for model_name, model in models.items():
        fold_results = []
        all_y_true = []
        all_y_prob = []
        all_y_pred = []

        for ds in PARKINSONS_DATASETS:
            test_mask = datasets == ds
            train_mask = ~test_mask

            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            pipe = make_pipeline(model.__class__(**model.get_params()))
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics["dataset_id"] = ds
            metrics["n_test"] = int(test_mask.sum())
            metrics["n_train"] = int(train_mask.sum())
            fold_results.append(metrics)

            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)
            all_y_pred.extend(y_pred)

        # Aggregate
        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_y_pred = np.array(all_y_pred)

        agg = compute_metrics(all_y_true, all_y_pred, all_y_prob)

        acc_vals = [f["accuracy"] for f in fold_results]
        auc_vals = [f["auc"] for f in fold_results]

        summary = {
            "aggregate": agg,
            "per_fold": fold_results,
            "accuracy_mean": float(np.mean(acc_vals)),
            "accuracy_std": float(np.std(acc_vals)),
            "auc_mean": float(np.mean(auc_vals)),
            "auc_std": float(np.std(auc_vals)),
        }
        results[model_name] = summary

        print(f"  {model_name}: "
              f"Acc={agg['accuracy']:.3f} (mean={np.mean(acc_vals):.3f}±{np.std(acc_vals):.3f}), "
              f"AUC={agg['auc']:.3f} (mean={np.mean(auc_vals):.3f}±{np.std(auc_vals):.3f})")

    return results


# ── Stratified k-fold CV ───────────────────────────────────────��─────────────

def run_stratified_cv(df_pk, feature_cols, label="stratified"):
    """Stratified 5-fold cross-validation (upper bound)."""
    print(f"\n--- Stratified 5-fold CV ({label}) ---")

    X = df_pk[feature_cols].values
    y = df_pk["label"].values

    models = {
        "logistic_regression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for model_name, model in models.items():
        fold_results = []
        all_y_true = []
        all_y_prob = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = make_pipeline(model.__class__(**model.get_params()))
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics["fold"] = fold_idx
            metrics["n_test"] = len(test_idx)
            fold_results.append(metrics)

            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)
            all_y_pred.extend(y_pred)

        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_y_pred = np.array(all_y_pred)

        agg = compute_metrics(all_y_true, all_y_pred, all_y_prob)

        acc_vals = [f["accuracy"] for f in fold_results]
        auc_vals = [f["auc"] for f in fold_results]

        summary = {
            "aggregate": agg,
            "per_fold": fold_results,
            "accuracy_mean": float(np.mean(acc_vals)),
            "accuracy_std": float(np.std(acc_vals)),
            "auc_mean": float(np.mean(auc_vals)),
            "auc_std": float(np.std(auc_vals)),
        }
        results[model_name] = summary

        print(f"  {model_name}: "
              f"Acc={agg['accuracy']:.3f} (mean={np.mean(acc_vals):.3f}±{np.std(acc_vals):.3f}), "
              f"AUC={agg['auc']:.3f} (mean={np.mean(auc_vals):.3f}±{np.std(auc_vals):.3f})")

    return results


# ── Age confound analysis ───────────────────────────────���────────────────────

def run_age_confound(df_pk, stats_out):
    """Run LODO with and without age to detect age confound."""
    print("\n=== Age confound analysis ===")

    # With age
    df_with_age = df_pk.dropna(subset=["age_years"]).copy()
    feature_cols_with_age = FEATURE_COLS + ["age_years"]

    if len(df_with_age) < 50:
        print("  Too few subjects with age, skipping age confound analysis")
        stats_out["age_confound"] = {"skipped": True}
        return

    print(f"  Subjects with age: {len(df_with_age)}")

    results_without_age = run_lodo_cv(df_with_age, FEATURE_COLS, label="without_age")
    results_with_age = run_lodo_cv(df_with_age, feature_cols_with_age, label="with_age")

    stats_out["age_confound"] = {
        "n_subjects": len(df_with_age),
        "without_age": results_without_age,
        "with_age": results_with_age,
    }

    # Report
    for model in ["logistic_regression", "random_forest"]:
        auc_without = results_without_age[model]["aggregate"]["auc"]
        auc_with = results_with_age[model]["aggregate"]["auc"]
        print(f"  {model}: AUC without age = {auc_without:.3f}, "
              f"with age = {auc_with:.3f}, delta = {auc_with - auc_without:+.3f}")


# ── Dataset confound analysis ──────────────────────────────────────��─────────

def _residualize_in_fold(X_train, X_test, ds_train, ds_test, all_datasets):
    """Residualize features by regressing out dataset identity.

    Fits OLS on training data only, applies to both train and test.
    Returns (X_train_resid, X_test_resid).
    """
    # Build dummy matrices with consistent column ordering
    def _dummies(ds_arr):
        out = np.zeros((len(ds_arr), len(all_datasets)), dtype=float)
        ds_to_idx = {d: i for i, d in enumerate(all_datasets)}
        for row, d in enumerate(ds_arr):
            out[row, ds_to_idx[d]] = 1.0
        return out

    D_train = _dummies(ds_train)
    D_test = _dummies(ds_test)

    X_train_resid = np.zeros_like(X_train)
    X_test_resid = np.zeros_like(X_test)

    for j in range(X_train.shape[1]):
        reg = LinearRegression()
        reg.fit(D_train, X_train[:, j])
        X_train_resid[:, j] = X_train[:, j] - reg.predict(D_train)
        X_test_resid[:, j] = X_test[:, j] - reg.predict(D_test)

    return X_train_resid, X_test_resid


def run_lodo_cv_residualized(df_pk, feature_cols, label="residualized"):
    """LODO CV with dataset-identity residualization inside each fold.

    For each fold, fits OLS on training subjects only, then applies
    residualization to both training and test subjects. No data leakage.
    """
    print(f"\n--- LODO CV ({label}, {len(feature_cols)} features, "
          f"in-fold residualization) ---")

    X = df_pk[feature_cols].values
    y = df_pk["label"].values
    datasets = df_pk["dataset_id"].values
    all_datasets = sorted(df_pk["dataset_id"].unique())

    # Impute once (median from full data is OK — it's a per-column
    # constant, not learned from labels or fold structure)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    models = {
        "logistic_regression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = {}
    for model_name, model in models.items():
        fold_results = []
        all_y_true = []
        all_y_prob = []
        all_y_pred = []

        for ds in PARKINSONS_DATASETS:
            test_mask = datasets == ds
            train_mask = ~test_mask

            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue

            X_train_imp = X_imputed[train_mask]
            X_test_imp = X_imputed[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            ds_train = datasets[train_mask]
            ds_test = datasets[test_mask]

            # Residualize using only training data
            X_train_r, X_test_r = _residualize_in_fold(
                X_train_imp, X_test_imp, ds_train, ds_test, all_datasets,
            )

            # Scale and classify (no imputer needed — already imputed)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", model.__class__(**model.get_params())),
            ])
            pipe.fit(X_train_r, y_train)

            y_pred = pipe.predict(X_test_r)
            y_prob = pipe.predict_proba(X_test_r)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics["dataset_id"] = ds
            metrics["n_test"] = int(test_mask.sum())
            metrics["n_train"] = int(train_mask.sum())
            fold_results.append(metrics)

            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)
            all_y_pred.extend(y_pred)

        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_y_pred = np.array(all_y_pred)

        agg = compute_metrics(all_y_true, all_y_pred, all_y_prob)

        acc_vals = [f["accuracy"] for f in fold_results]
        auc_vals = [f["auc"] for f in fold_results]

        summary = {
            "aggregate": agg,
            "per_fold": fold_results,
            "accuracy_mean": float(np.mean(acc_vals)),
            "accuracy_std": float(np.std(acc_vals)),
            "auc_mean": float(np.mean(auc_vals)),
            "auc_std": float(np.std(auc_vals)),
        }
        results[model_name] = summary

        print(f"  {model_name}: "
              f"Acc={agg['accuracy']:.3f} (mean={np.mean(acc_vals):.3f}"
              f"±{np.std(acc_vals):.3f}), "
              f"AUC={agg['auc']:.3f} (mean={np.mean(auc_vals):.3f}"
              f"±{np.std(auc_vals):.3f})")

    return results


def run_dataset_confound(df_pk, stats_out):
    """Test whether spectral features encode dataset identity."""
    print("\n=== Dataset confound analysis ===")

    X = df_pk[FEATURE_COLS].values
    y_ds = df_pk["dataset_id"].values

    # 1. Dataset identity classifier (LODO-style: predict dataset from features)
    unique_ds = sorted(df_pk["dataset_id"].unique())
    ds_to_int = {ds: i for i, ds in enumerate(unique_ds)}
    y_ds_int = np.array([ds_to_int[d] for d in y_ds])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ds_preds = np.zeros_like(y_ds_int)

    for train_idx, test_idx in skf.split(X, y_ds_int):
        pipe_clone = make_pipeline(LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            multi_class="multinomial", random_state=RANDOM_STATE,
        ))
        pipe_clone.fit(X[train_idx], y_ds_int[train_idx])
        ds_preds[test_idx] = pipe_clone.predict(X[test_idx])

    ds_accuracy = accuracy_score(y_ds_int, ds_preds)
    chance = 1.0 / len(unique_ds)
    print(f"  Dataset identity accuracy: {ds_accuracy:.3f} (chance = {chance:.3f})")

    # 2. Residualized LODO — regression fitted inside each fold (no leakage)
    results_residualized = run_lodo_cv_residualized(
        df_pk, FEATURE_COLS, label="residualized",
    )

    # 3. Raw LODO for comparison
    results_raw = run_lodo_cv(df_pk, FEATURE_COLS, label="raw")

    stats_out["dataset_confound"] = {
        "dataset_identity_accuracy": float(ds_accuracy),
        "dataset_identity_chance": float(chance),
        "raw_lodo": results_raw,
        "residualized_lodo": results_residualized,
    }

    # Save residualized per-fold CSV
    resid_rows = []
    for model_name, model_results in results_residualized.items():
        for fold in model_results["per_fold"]:
            row = {"model": model_name}
            row.update(fold)
            resid_rows.append(row)
    pd.DataFrame(resid_rows).to_csv(
        EXP2_DIR / "lodo_cv_results_residualized_fixed.csv", index=False,
    )
    print(f"  Saved lodo_cv_results_residualized_fixed.csv")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model_name in zip(axes, ["logistic_regression", "random_forest"]):
        raw_auc = results_raw[model_name]["aggregate"]["auc"]
        resid_auc = results_residualized[model_name]["aggregate"]["auc"]

        raw_per = [f["auc"] for f in results_raw[model_name]["per_fold"]]
        resid_per = [f["auc"] for f in results_residualized[model_name]["per_fold"]]

        x = np.arange(len(PARKINSONS_DATASETS))
        width = 0.35

        ax.bar(x - width/2, raw_per, width, label=f"Raw (AUC={raw_auc:.3f})",
               color="steelblue", alpha=0.8)
        ax.bar(x + width/2, resid_per, width, label=f"Residualized (AUC={resid_auc:.3f})",
               color="coral", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(PARKINSONS_DATASETS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("AUC")
        ax.set_title(model_name.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle(f"Dataset confound: raw vs residualized LODO classification\n"
                 f"Dataset identity accuracy: {ds_accuracy:.1%} (chance = {chance:.1%})",
                 fontsize=13)
    fig.subplots_adjust(top=0.85)

    fig.savefig(EXP2_DIR / "dataset_confound.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved dataset_confound.png")


# ── Feature importance ───────────────────────────────────────────────────────

def run_feature_importance(df_pk, stats_out):
    """Train on all data, extract feature importances."""
    print("\n=== Feature importance ===")

    X = df_pk[FEATURE_COLS].values
    y = df_pk["label"].values

    # Logistic regression
    pipe_lr = make_pipeline(LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=RANDOM_STATE,
    ))
    pipe_lr.fit(X, y)
    lr_coefs = np.abs(pipe_lr.named_steps["clf"].coef_[0])

    # Random forest
    pipe_rf = make_pipeline(RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ))
    pipe_rf.fit(X, y)
    rf_importances = pipe_rf.named_steps["clf"].feature_importances_

    # Map to feature names
    feature_names = FEATURE_COLS
    lr_imp = list(zip(feature_names, lr_coefs))
    rf_imp = list(zip(feature_names, rf_importances))

    lr_imp.sort(key=lambda x: x[1], reverse=True)
    rf_imp.sort(key=lambda x: x[1], reverse=True)

    stats_out["feature_importance"] = {
        "logistic_regression_top20": [
            {"feature": f, "abs_coef": float(v)} for f, v in lr_imp[:20]
        ],
        "random_forest_top20": [
            {"feature": f, "importance": float(v)} for f, v in rf_imp[:20]
        ],
    }

    print("  Logistic regression top 10:")
    for f, v in lr_imp[:10]:
        print(f"    {f:25s} |coef| = {v:.4f}")
    print("  Random forest top 10:")
    for f, v in rf_imp[:10]:
        print(f"    {f:25s} importance = {v:.4f}")

    # Plot top-20 features side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # LR
    top20_lr = lr_imp[:20][::-1]
    ax1.barh([f[0] for f in top20_lr], [f[1] for f in top20_lr], color="steelblue")
    ax1.set_xlabel("|Coefficient|")
    ax1.set_title("Logistic Regression\n(absolute coefficients, standardized)")
    ax1.tick_params(axis="y", labelsize=9)

    # RF
    top20_rf = rf_imp[:20][::-1]
    ax2.barh([f[0] for f in top20_rf], [f[1] for f in top20_rf], color="forestgreen")
    ax2.set_xlabel("Feature importance (MDI)")
    ax2.set_title("Random Forest\n(mean decrease in impurity)")
    ax2.tick_params(axis="y", labelsize=9)

    fig.suptitle("Top 20 features for Parkinson's vs control classification", fontsize=14)
    fig.tight_layout()

    fig.savefig(EXP2_DIR / "feature_importance.png")
    plt.close(fig)
    print(f"  Saved feature_importance.png")


# ── Plots: confusion matrix, ROC ─────────────────────────────────────────────

def plot_confusion_matrix(lodo_results, stats_out):
    """Plot confusion matrices for LODO CV."""
    print("\n=== Plotting confusion matrices ===")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model_name in zip(axes, ["logistic_regression", "random_forest"]):
        agg = lodo_results[model_name]["aggregate"]
        cm = np.array([[agg["tn"], agg["fp"]], [agg["fn"], agg["tp"]]])

        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=16, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Control", "Parkinson's"])
        ax.set_yticklabels(["Control", "Parkinson's"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        title = model_name.replace("_", " ").title()
        ax.set_title(f"{title}\n"
                     f"Acc={agg['accuracy']:.3f}, AUC={agg['auc']:.3f}, "
                     f"Sens={agg['sensitivity']:.3f}, Spec={agg['specificity']:.3f}")

    fig.suptitle("LODO Cross-Validation: Confusion Matrices", fontsize=14)
    fig.tight_layout()

    fig.savefig(EXP2_DIR / "confusion_matrix_lodo.png")
    plt.close(fig)
    print(f"  Saved confusion_matrix_lodo.png")


def plot_roc_curves(df_pk, stats_out):
    """Plot ROC curves with per-fold and mean curves."""
    print("\n=== Plotting ROC curves ===")

    X = df_pk[FEATURE_COLS].values
    y = df_pk["label"].values
    datasets = df_pk["dataset_id"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (model_name, model_cls, model_kwargs) in zip(axes, [
        ("Logistic Regression", LogisticRegression,
         dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
              class_weight="balanced", random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier,
         dict(n_estimators=500, class_weight="balanced",
              random_state=RANDOM_STATE, n_jobs=-1)),
    ]):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for ds in PARKINSONS_DATASETS:
            test_mask = datasets == ds
            train_mask = ~test_mask
            if test_mask.sum() == 0:
                continue

            pipe = make_pipeline(model_cls(**model_kwargs))
            pipe.fit(X[train_mask], y[train_mask])
            y_prob = pipe.predict_proba(X[test_mask])[:, 1]

            fpr, tpr, _ = roc_curve(y[test_mask], y_prob)
            auc_val = roc_auc_score(y[test_mask], y_prob)
            aucs.append(auc_val)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            ax.plot(fpr, tpr, alpha=0.3, linewidth=1,
                    label=f"{ds} (AUC={auc_val:.2f})")

        # Mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, "b-", linewidth=2.5,
                label=f"Mean (AUC={mean_auc:.2f}±{std_auc:.2f})")

        # ±1 SD band
        std_tpr = np.std(tprs, axis=0)
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        alpha=0.15, color="blue")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(model_name)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("LODO Cross-Validation: ROC Curves\n"
                 "Parkinson's vs Neurotypical", fontsize=14)
    fig.tight_layout()

    fig.savefig(EXP2_DIR / "roc_curve_lodo.png")
    plt.close(fig)
    print(f"  Saved roc_curve_lodo.png")


# ── Pre-analysis report ─────────────────────────────────��───────────────────���

def pre_analysis_report(df):
    """Report group distribution for classification."""
    print("=" * 70)
    print("EXPERIMENT 2: PRE-ANALYSIS REPORT")
    print("=" * 70)

    df_labeled = df[df["group_label"].notna()].copy()
    print(f"\nSubjects with group labels: {len(df_labeled)}")
    print("\nGroup distribution (all):")
    groups = df_labeled.groupby(["group_label", "group_role"]).size().sort_values(ascending=False)
    for (label, role), n in groups.items():
        ds_count = df_labeled[(df_labeled["group_label"] == label)]["dataset_id"].nunique()
        marker = " ***" if n >= 30 else ""
        print(f"  {label:35s} ({role:10s}): {n:5d} in {ds_count} datasets{marker}")

    # Parkinson's specific
    pk = df[
        df["dataset_id"].isin(PARKINSONS_DATASETS) &
        df["group_role"].isin(["patient", "control"])
    ]
    print(f"\nParkinson's datasets subjects in preprocessed data: {len(pk)}")
    n_with_age = pk["age_years"].notna().sum()
    print(f"  With age: {n_with_age}")
    if n_with_age > 0:
        pat_age = pk[pk["group_role"] == "patient"]["age_years"].dropna()
        ctrl_age = pk[pk["group_role"] == "control"]["age_years"].dropna()
        print(f"  Patient age: {pat_age.mean():.1f} ± {pat_age.std():.1f} (n={len(pat_age)})")
        print(f"  Control age: {ctrl_age.mean():.1f} ± {ctrl_age.std():.1f} (n={len(ctrl_age)})")

    print("=" * 70)


# ── Sensitivity analysis: frontal channel removal ───────────────────────────

def run_frontal_removal(df_pk, output_dir):
    """LODO with Fp1, Fp2, Fz features removed (18 dropped, 96 remaining)."""
    print("\n=== Sensitivity: frontal channel removal ===")
    frontal = {"Fp1", "Fp2", "Fz"}
    reduced = [c for c in FEATURE_COLS
               if c.split("_")[-1] not in frontal
               and c.replace("peak_alpha_", "") not in frontal]
    print(f"  Removed {len(FEATURE_COLS) - len(reduced)} features, {len(reduced)} remaining")
    results = run_lodo_cv(df_pk, reduced, label="frontal_removed")
    _save_lodo_csv(results, output_dir / "frontal_removed_results.csv")
    return results


# ── Sensitivity analysis: gamma-band removal ────────────────────────────────

def run_gamma_removal(df_pk, output_dir):
    """LODO with all gamma-band features removed (19 dropped, 95 remaining)."""
    print("\n=== Sensitivity: gamma-band removal ===")
    reduced = [c for c in FEATURE_COLS if not c.startswith("gamma_")]
    print(f"  Removed {len(FEATURE_COLS) - len(reduced)} features, {len(reduced)} remaining")
    results = run_lodo_cv(df_pk, reduced, label="gamma_removed")
    _save_lodo_csv(results, output_dir / "gamma_removed_results.csv")
    return results


# ── Sensitivity analysis: MoCA cognitive covariate ──────────────────────────

MOCA_DATASETS = ["ds004574", "ds004579", "ds004580", "ds004584"]


def run_moca_covariate(df_pk, catalog_db, output_dir):
    """LODO with MoCA residualization on the 4 datasets that have MoCA scores.

    Three conditions: spectral only, spectral + MoCA, spectral residualized
    against MoCA (OLS fitted inside each fold, no leakage).
    """
    print("\n=== Sensitivity: MoCA cognitive covariate ===")

    db = sqlite3.connect(str(catalog_db))
    moca = pd.read_sql_query(
        "SELECT dataset_id, participant_id, score_moca FROM participants "
        "WHERE score_moca IS NOT NULL AND dataset_id IN ({})".format(
            ",".join(f"'{d}'" for d in MOCA_DATASETS)
        ), db,
    )
    db.close()

    df_m = df_pk.merge(moca, on=["dataset_id", "participant_id"], how="inner")
    df_m = df_m[df_m["group_role"].isin(["patient", "control"])].copy()
    df_m["label"] = (df_m["group_role"] == "patient").astype(int)
    print(f"  Subjects with MoCA: {len(df_m)} "
          f"({(df_m['label']==1).sum()} PD, {(df_m['label']==0).sum()} NT)")

    # Condition 1: spectral only
    res_spectral = _lodo_4ds(df_m, FEATURE_COLS, "spectral_only")
    # Condition 2: spectral + MoCA
    res_plus = _lodo_4ds(df_m, FEATURE_COLS + ["score_moca"], "spectral+moca")
    # Condition 3: spectral residualized against MoCA
    res_resid = _lodo_4ds_moca_resid(df_m, FEATURE_COLS)

    # Save combined CSV
    rows = []
    for cond, res in [("spectral_only", res_spectral),
                      ("spectral_plus_moca", res_plus),
                      ("spectral_resid_moca", res_resid)]:
        for model_name, model_res in res.items():
            for fold in model_res["per_fold"]:
                row = {"condition": cond, "model": model_name}
                row.update(fold)
                rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "moca_covariate_results.csv", index=False)
    print(f"  Saved moca_covariate_results.csv")

    return {"spectral_only": res_spectral, "spectral_plus_moca": res_plus,
            "spectral_resid_moca": res_resid}


def _lodo_4ds(df_in, feature_cols, label):
    """LODO across the 4 MoCA datasets only."""
    X = df_in[feature_cols].values
    y = df_in["label"].values
    datasets = df_in["dataset_id"].values
    models = _make_models()
    results = {}
    for model_name, model in models.items():
        fold_results, all_yt, all_yp, all_ypred = [], [], [], []
        for ds in MOCA_DATASETS:
            test_mask = datasets == ds
            train_mask = ~test_mask
            pipe = make_pipeline(model.__class__(**model.get_params()))
            pipe.fit(X[train_mask], y[train_mask])
            y_pred = pipe.predict(X[test_mask])
            y_prob = pipe.predict_proba(X[test_mask])[:, 1]
            m = compute_metrics(y[test_mask], y_pred, y_prob)
            m["dataset_id"] = ds
            m["n_test"] = int(test_mask.sum())
            fold_results.append(m)
            all_yt.extend(y[test_mask])
            all_yp.extend(y_prob)
            all_ypred.extend(y_pred)
        agg = compute_metrics(np.array(all_yt), np.array(all_ypred), np.array(all_yp))
        results[model_name] = {"aggregate": agg, "per_fold": fold_results}
    return results


def _lodo_4ds_moca_resid(df_in, feature_cols):
    """LODO across 4 MoCA datasets with in-fold MoCA residualization."""
    X = df_in[feature_cols].values
    y = df_in["label"].values
    moca_vals = df_in["score_moca"].values.reshape(-1, 1)
    datasets = df_in["dataset_id"].values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    models = _make_models()
    results = {}
    for model_name, model in models.items():
        fold_results, all_yt, all_yp, all_ypred = [], [], [], []
        for ds in MOCA_DATASETS:
            test_mask = datasets == ds
            train_mask = ~test_mask
            X_tr, X_te = X_imp[train_mask], X_imp[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            m_tr, m_te = moca_vals[train_mask], moca_vals[test_mask]
            X_tr_r = np.zeros_like(X_tr)
            X_te_r = np.zeros_like(X_te)
            for j in range(X_tr.shape[1]):
                reg = LinearRegression()
                reg.fit(m_tr, X_tr[:, j])
                X_tr_r[:, j] = X_tr[:, j] - reg.predict(m_tr)
                X_te_r[:, j] = X_te[:, j] - reg.predict(m_te)
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", model.__class__(**model.get_params())),
            ])
            pipe.fit(X_tr_r, y_tr)
            y_pred = pipe.predict(X_te_r)
            y_prob = pipe.predict_proba(X_te_r)[:, 1]
            m = compute_metrics(y_te, y_pred, y_prob)
            m["dataset_id"] = ds
            m["n_test"] = int(test_mask.sum())
            fold_results.append(m)
            all_yt.extend(y_te)
            all_yp.extend(y_prob)
            all_ypred.extend(y_pred)
        agg = compute_metrics(np.array(all_yt), np.array(all_ypred), np.array(all_yp))
        results[model_name] = {"aggregate": agg, "per_fold": fold_results}
    return results


# ── Sensitivity analysis: label permutation test ────────────────────────────

N_PERMUTATIONS = 100


def run_permutation_test(df_pk, output_dir):
    """Within-dataset label permutation test (100 permutations)."""
    print(f"\n=== Sensitivity: label permutation test ({N_PERMUTATIONS} permutations) ===")
    X = df_pk[FEATURE_COLS].values
    y = df_pk["label"].values
    datasets = df_pk["dataset_id"].values
    models = _make_models()

    # Real AUC
    real_aucs = {}
    for model_name, model in models.items():
        all_yt, all_yp = [], []
        for ds in PARKINSONS_DATASETS:
            test_mask = datasets == ds
            train_mask = ~test_mask
            pipe = make_pipeline(model.__class__(**model.get_params()))
            pipe.fit(X[train_mask], y[train_mask])
            prob = pipe.predict_proba(X[test_mask])[:, 1]
            all_yt.extend(y[test_mask])
            all_yp.extend(prob)
        real_aucs[model_name] = roc_auc_score(all_yt, all_yp)
    print(f"  Real AUC — LR: {real_aucs['logistic_regression']:.4f}, "
          f"RF: {real_aucs['random_forest']:.4f}")

    # Permutations
    rng = np.random.RandomState(RANDOM_STATE)
    perm_aucs = {mn: [] for mn in models}

    for i in range(N_PERMUTATIONS):
        y_perm = y.copy()
        for ds in PARKINSONS_DATASETS:
            mask = datasets == ds
            y_perm[mask] = rng.permutation(y_perm[mask])
        for model_name, model in models.items():
            all_yt, all_yp = [], []
            for ds in PARKINSONS_DATASETS:
                test_mask = datasets == ds
                train_mask = ~test_mask
                pipe = make_pipeline(model.__class__(**model.get_params()))
                pipe.fit(X[train_mask], y_perm[train_mask])
                prob = pipe.predict_proba(X[test_mask])[:, 1]
                all_yt.extend(y_perm[test_mask])
                all_yp.extend(prob)
            perm_aucs[model_name].append(roc_auc_score(all_yt, all_yp))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_PERMUTATIONS} done")

    # Save CSV
    rows = [{"permutation": i + 1,
             "logistic_regression_auc": perm_aucs["logistic_regression"][i],
             "random_forest_auc": perm_aucs["random_forest"][i]}
            for i in range(N_PERMUTATIONS)]
    pd.DataFrame(rows).to_csv(output_dir / "permutation_test_results.csv", index=False)
    print(f"  Saved permutation_test_results.csv")

    # Report
    for mn in ["logistic_regression", "random_forest"]:
        arr = np.array(perm_aucs[mn])
        real = real_aucs[mn]
        p = (np.sum(arr >= real) + 1) / (N_PERMUTATIONS + 1)
        print(f"  {mn}: real={real:.4f}, perm mean={arr.mean():.4f}±{arr.std():.4f}, "
              f"p={p:.4f}")

    return {"real_aucs": real_aucs, "perm_aucs": perm_aucs}


# ── Shared helpers ──────────────────────────────────────────────────────────

def _make_models():
    return {
        "logistic_regression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def _save_lodo_csv(results, path):
    rows = []
    for model_name, model_res in results.items():
        for fold in model_res["per_fold"]:
            row = {"model": model_name}
            row.update(fold)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved {path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PD vs control cross-dataset classification (Experiment 2).",
    )
    parser.add_argument(
        "features_csv", type=Path,
        help="Path to features.csv (subject-level spectral features).",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory. Defaults to 'experiment2/' next to features_csv.",
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run sensitivity analyses (frontal removal, gamma removal, "
             "MoCA covariate, label permutation).",
    )
    parser.add_argument(
        "--catalog-db", type=Path, default=None,
        help="Path to catalog.db (required for MoCA analysis). "
             "Defaults to ../../catalog.db relative to features_csv.",
    )
    args = parser.parse_args()

    features_csv = args.features_csv.resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = features_csv.parent / "experiment2"
    output_dir.mkdir(parents=True, exist_ok=True)

    global EXP2_DIR
    EXP2_DIR = output_dir

    print("Loading features.csv...")
    df = pd.read_csv(features_csv)
    print(f"  Loaded {len(df)} subjects")

    pre_analysis_report(df)

    stats_out = {}

    # Prepare Parkinson's data
    df_pk = prepare_parkinsons_data(df)
    if len(df_pk) < 50:
        print("ERROR: Too few subjects for classification. Aborting.")
        return

    stats_out["n_subjects"] = len(df_pk)
    stats_out["n_patients"] = int((df_pk["label"] == 1).sum())
    stats_out["n_controls"] = int((df_pk["label"] == 0).sum())

    # 1. LODO CV
    print("\n" + "=" * 70)
    print("LEAVE-ONE-DATASET-OUT CROSS-VALIDATION")
    print("=" * 70)
    lodo_results = run_lodo_cv(df_pk, FEATURE_COLS)
    stats_out["lodo"] = lodo_results
    _save_lodo_csv(lodo_results, output_dir / "lodo_cv_results.csv")

    # 2. Stratified 5-fold CV
    print("\n" + "=" * 70)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION (upper bound)")
    print("=" * 70)
    strat_results = run_stratified_cv(df_pk, FEATURE_COLS)
    stats_out["stratified"] = strat_results

    strat_rows = []
    for model_name, model_results in strat_results.items():
        for fold in model_results["per_fold"]:
            row = {"model": model_name}
            row.update(fold)
            strat_rows.append(row)
    pd.DataFrame(strat_rows).to_csv(output_dir / "stratified_cv_results.csv", index=False)

    # 3. Age confound
    print("\n" + "=" * 70)
    print("AGE CONFOUND ANALYSIS")
    print("=" * 70)
    run_age_confound(df_pk, stats_out)

    # 4. Dataset confound
    print("\n" + "=" * 70)
    print("DATASET CONFOUND ANALYSIS")
    print("=" * 70)
    run_dataset_confound(df_pk, stats_out)

    # 5. Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    run_feature_importance(df_pk, stats_out)

    # 6. Plots
    plot_confusion_matrix(lodo_results, stats_out)
    plot_roc_curves(df_pk, stats_out)

    # 7. Sensitivity analyses (optional)
    if args.sensitivity:
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSES")
        print("=" * 70)

        run_frontal_removal(df_pk, output_dir)
        run_gamma_removal(df_pk, output_dir)

        catalog_db = args.catalog_db
        if catalog_db is None:
            catalog_db = features_csv.parent.parent / "catalog.db"
        if catalog_db.exists():
            run_moca_covariate(df_pk, catalog_db, output_dir)
        else:
            print(f"\n  Skipping MoCA analysis: catalog.db not found at {catalog_db}")

        run_permutation_test(df_pk, output_dir)

    # Save stats
    with open(output_dir / "experiment2_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2, default=str)
    print(f"\nSaved experiment2_stats.json")
    print("\nExperiment 2 complete!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
