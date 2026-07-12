"""
09_ml_training_and_evaluation.py
--------------------------------------------------------------------
Run after:
  01_data_cleaning.py
  02_feature_engineering.py
  03_exploratory_data_analysis.py
  04_ship_mode_analysis.py
  05_state_analysis.py
  06_region_analysis.py
  07_factory_analysis.py
  08_streamlit_dashboard.py

Input : featured_nassau_candy.csv   (output of 02_feature_engineering.py)
Output: ML_Models/                  (trained models + train/test data)
        ML_Charts/                 (all evaluation visualizations)
        ML_Reports/                (all evaluation CSVs + final text report)

--------------------------------------------------------------------
TARGET DEFINITION (decided explicitly, so there is only ONE version
of "the truth" instead of the two conflicting ones the old files used)
--------------------------------------------------------------------
Classification target : "Delay Status"  (3 classes: On Time / Moderate
                         Delay / Delayed) - kept as the full multi-class
                         business label instead of collapsing it to a
                         binary "Is Delayed" flag, since the project's
                         Delay Status / Route Efficiency phases are
                         built around all three categories.
Regression target      : "Shipping Lead Time (Simulated)" (days)

Both targets are derived from the SIMULATED shipping lead time
(see 01_data_cleaning.py headline finding: the real Ship Date column
is corrupted for 100% of records). This is clearly labelled throughout
this script's charts, reports and recommendations.
"""

import os
import json
import pickle
import textwrap
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import joblib
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score,
    learning_curve, validation_curve,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "figure.figsize": (12, 6),
    "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10, "font.family": "DejaVu Sans",
})

PALETTE_CAT = "Set2"
ACCENT_COLOR = "#2C7BB6"
GREEN_COLOR = "#27AE60"
RED_COLOR = "#E74C3C"
ORANGE_COLOR = "#E67E22"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATA_PATH = "featured_nassau_candy.csv"
MODELS_DIR = "ML_Models"
CHARTS_DIR = "ML_Charts"
REPORTS_DIR = "ML_Reports"
for d in [MODELS_DIR, CHARTS_DIR, REPORTS_DIR, os.path.join(MODELS_DIR, "data")]:
    os.makedirs(d, exist_ok=True)

LEAD_TIME_COL = "Shipping Lead Time (Simulated)"
DELAY_COL = "Delay Status"
MARGIN_COL = "Profit Margin %"

CATEGORICAL_FEATURES = ["Ship Mode", "Region", "Factory", "Division"]
NUMERIC_FEATURES = ["Sales", "Units", "Cost", "Gross Profit", MARGIN_COL,
                    "Order Month Number", "Order Quarter", "Is US Record"]

CLASSIFICATION_MODEL_FACTORY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "decision_tree": lambda: DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=RANDOM_STATE),
}
REGRESSION_MODEL_FACTORY = {
    "linear_regression": lambda: LinearRegression(),
    "decision_tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=RANDOM_STATE),
}

# ─────────────────────────────────────────────────────────────
# COUNTERS / ACCUMULATORS
# ─────────────────────────────────────────────────────────────
chart_count = 0
summary_count = 0
report_count = 0
insight_list = []
CV_FOLD_SCORES_CACHE = {}


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def save_chart(filename: str) -> None:
    global chart_count
    path = os.path.join(CHARTS_DIR, filename)
    try:
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        print(f"  WARNING: could not save chart '{filename}': {exc}")
    finally:
        plt.close("all")
    chart_count += 1
    print(f"  [Chart saved]   {path}")


def save_summary(data: pd.DataFrame, filename: str) -> None:
    global summary_count
    path = os.path.join(REPORTS_DIR, filename)
    try:
        data.to_csv(path, index=False, float_format="%.4f")
        summary_count += 1
        print(f"  [Summary saved] {path}")
    except Exception as exc:
        print(f"  WARNING: could not save summary '{filename}': {exc}")


def save_report_text(text: str, filename: str) -> None:
    global report_count
    path = os.path.join(REPORTS_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        report_count += 1
        print(f"  [Report saved]  {path}")
    except OSError as exc:
        print(f"  WARNING: could not save report '{filename}': {exc}")


def add_insight(text: str) -> None:
    insight_list.append(text)
    print(f"  [Insight {len(insight_list):02d}] {text}")


def section(title: str) -> str:
    border = "=" * 70
    return f"\n{border}\n{title}\n{border}\n"


def save_model(model, name: str, kind: str) -> None:
    path = os.path.join(MODELS_DIR, f"{kind}_{name}.joblib")
    try:
        joblib.dump(model, path)
    except Exception as exc:
        print(f"  WARNING: failed to save model '{name}': {exc}")


# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATASET
# ════════════════════════════════════════════════════════════════

def step1_load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    print(section("STEP 1: Load Dataset"))
    try:
        data = pd.read_csv(path, low_memory=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"'{path}' not found. Run 01_data_cleaning.py and 02_feature_engineering.py first."
        ) from exc

    required = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [DELAY_COL, LEAD_TIME_COL]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(
            f"Required column(s) missing from '{path}': {missing}. "
            "Re-run 02_feature_engineering.py to regenerate the featured dataset."
        )

    data = data.dropna(subset=[DELAY_COL, LEAD_TIME_COL]).reset_index(drop=True)

    print(f"Dataset Shape          : {data.shape}")
    print(f"Classification target  : '{DELAY_COL}' -> {sorted(data[DELAY_COL].unique())}")
    print(f"Regression target      : '{LEAD_TIME_COL}'")
    return data


# ════════════════════════════════════════════════════════════════
# STEP 2 — BUILD ML-READY FEATURE MATRIX
# ════════════════════════════════════════════════════════════════

def step2_build_feature_matrix(data: pd.DataFrame):
    print(section("STEP 2: Build ML-Ready Feature Matrix"))

    work = data.copy()
    work["Is US Record"] = work["Is US Record"].astype(int)

    dummies = pd.get_dummies(work[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES, drop_first=True)
    X = pd.concat([work[NUMERIC_FEATURES].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    X = X.fillna(X.median(numeric_only=True))

    y_class = work[DELAY_COL].astype(str)
    y_reg = work[LEAD_TIME_COL].astype(float)

    print(f"Feature matrix shape   : {X.shape}")
    print(f"Classification classes : {sorted(y_class.unique())}")
    print(f"Class balance:\n{y_class.value_counts(normalize=True).round(3).to_string()}")
    return X, y_class, y_reg


# ════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN/TEST SPLIT + TRAIN ALL MODELS (single contract)
# ════════════════════════════════════════════════════════════════

def step3_train_models(X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series):
    print(section("STEP 3: Train/Test Split & Model Training"))

    # One shared split (by row) so classification and regression are
    # evaluated on the exact same held-out shipments.
    X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
        X, y_class, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    save_model(scaler, "feature_scaler", "preprocessing")

    classification_models = {}
    for name, factory in CLASSIFICATION_MODEL_FACTORY.items():
        model = factory()
        model.fit(X_train_scaled, y_train_c)
        classification_models[name] = model
        save_model(model, name, "classification")

    regression_models = {}
    for name, factory in REGRESSION_MODEL_FACTORY.items():
        model = factory()
        model.fit(X_train_scaled, y_train_r)
        regression_models[name] = model
        save_model(model, name, "regression")

    data_dir = os.path.join(MODELS_DIR, "data")
    X_train_scaled.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
    y_train_c.to_csv(os.path.join(data_dir, "y_train_class.csv"), index=False)
    y_test_c.to_csv(os.path.join(data_dir, "y_test_class.csv"), index=False)
    y_train_r.to_csv(os.path.join(data_dir, "y_train_reg.csv"), index=False)
    y_test_r.to_csv(os.path.join(data_dir, "y_test_reg.csv"), index=False)
    pd.DataFrame({"Feature": X.columns}).to_csv(os.path.join(MODELS_DIR, "feature_names.csv"), index=False)
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "classification_target": DELAY_COL, "regression_target": LEAD_TIME_COL,
            "classes": sorted(y_class.unique().tolist()), "feature_columns": X.columns.tolist(),
        }, fh, indent=2)

    print(f"  Trained {len(classification_models)} classification models "
         f"and {len(regression_models)} regression models -> '{MODELS_DIR}/'.")

    return (classification_models, regression_models,
            X_train_scaled, X_test_scaled, y_train_c, y_test_c, y_train_r, y_test_r)


# ════════════════════════════════════════════════════════════════
# STEP 4 — CLASSIFICATION METRICS
# ════════════════════════════════════════════════════════════════

def step4_classification_metrics(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    print(section("STEP 4: Classification Metrics (Train/Test Split)"))

    classes = sorted(y_test.unique().tolist())
    rows = []
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            kappa = cohen_kappa_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            train_acc = accuracy_score(y_train, y_pred_train)

            roc_auc_val, logloss = np.nan, np.nan
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                    y_test_bin = label_binarize(y_test, classes=classes)
                    roc_auc_val = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
                    logloss = log_loss(y_test, y_proba, labels=classes)
                except Exception as exc:
                    print(f"    NOTE: ROC-AUC/LogLoss unavailable for '{name}': {exc}")

            rows.append({
                "Model": name, "Accuracy": round(acc, 4), "Precision": round(prec, 4),
                "Recall": round(rec, 4), "F1_Score": round(f1, 4),
                "ROC_AUC": round(roc_auc_val, 4) if not np.isnan(roc_auc_val) else np.nan,
                "Log_Loss": round(logloss, 4) if not np.isnan(logloss) else np.nan,
                "Cohens_Kappa": round(kappa, 4), "MCC": round(mcc, 4),
                "Train_Accuracy": round(train_acc, 4), "Train_Test_Gap": round(train_acc - acc, 4),
            })
        except Exception as exc:
            print(f"  WARNING: metric computation failed for '{name}': {exc}")

    metrics_df = pd.DataFrame(rows).sort_values("F1_Score", ascending=False).reset_index(drop=True)
    save_summary(metrics_df, "classification_metrics.csv")
    print(metrics_df.to_string(index=False))
    return metrics_df


# ════════════════════════════════════════════════════════════════
# STEP 5 — REGRESSION METRICS
# ════════════════════════════════════════════════════════════════

def step5_regression_metrics(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    print(section("STEP 5: Regression Metrics (Train/Test Split)"))

    rows = []
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            nonzero_mask = y_test.abs() > 0.5  # MAPE undefined near zero (Same Day = 0-1 day lead time)
            mape = (
                mean_absolute_percentage_error(y_test[nonzero_mask], y_pred[nonzero_mask]) * 100
                if nonzero_mask.sum() > 0 else np.nan
            )
            train_r2 = r2_score(y_train, y_pred_train)

            rows.append({
                "Model": name, "MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4),
                "R2_Score": round(r2, 4), "MAPE_%": round(mape, 4),
                "Train_R2": round(train_r2, 4), "Train_Test_Gap": round(train_r2 - r2, 4),
            })
        except Exception as exc:
            print(f"  WARNING: metric computation failed for '{name}': {exc}")

    metrics_df = pd.DataFrame(rows).sort_values("R2_Score", ascending=False).reset_index(drop=True)
    save_summary(metrics_df, "regression_metrics.csv")
    print(metrics_df.to_string(index=False))
    return metrics_df


# ════════════════════════════════════════════════════════════════
# STEP 6 — CROSS VALIDATION
# ════════════════════════════════════════════════════════════════

def step6_cross_validation(clf_models: dict, reg_models: dict, X_full, y_class_full, y_reg_full) -> pd.DataFrame:
    print(section("STEP 6: Cross Validation (Stratified K-Fold / K-Fold)"))

    rows = []
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name, model in clf_models.items():
        try:
            scores = cross_val_score(model, X_full, y_class_full, cv=skf, scoring="f1_weighted", n_jobs=-1)
            CV_FOLD_SCORES_CACHE[("Classification", name)] = scores
            rows.append({"Task": "Classification", "Model": name, "CV_Method": "Stratified K-Fold",
                        "Scoring": "F1 (weighted)", "Mean_CV_Score": round(scores.mean(), 4),
                        "Std_Dev": round(scores.std(), 4), "Min_Score": round(scores.min(), 4),
                        "Max_Score": round(scores.max(), 4)})
        except Exception as exc:
            print(f"  WARNING: CV failed for classifier '{name}': {exc}")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name, model in reg_models.items():
        try:
            scores = cross_val_score(model, X_full, y_reg_full, cv=kf, scoring="r2", n_jobs=-1)
            CV_FOLD_SCORES_CACHE[("Regression", name)] = scores
            rows.append({"Task": "Regression", "Model": name, "CV_Method": "K-Fold", "Scoring": "R2",
                        "Mean_CV_Score": round(scores.mean(), 4), "Std_Dev": round(scores.std(), 4),
                        "Min_Score": round(scores.min(), 4), "Max_Score": round(scores.max(), 4)})
        except Exception as exc:
            print(f"  WARNING: CV failed for regressor '{name}': {exc}")

    cv_df = pd.DataFrame(rows)
    save_summary(cv_df, "cross_validation_results.csv")
    print(cv_df.to_string(index=False))
    return cv_df


# ════════════════════════════════════════════════════════════════
# STEP 7 — CONFUSION MATRICES
# ════════════════════════════════════════════════════════════════

def step7_confusion_matrices(clf_models: dict, X_test, y_test) -> pd.DataFrame:
    print(section("STEP 7: Confusion Matrices"))

    classes = sorted(y_test.unique().tolist())
    long_rows = []
    n_models = len(clf_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, clf_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                   yticklabels=classes, ax=ax, cbar=False)
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i, true_label in enumerate(classes):
            for j, pred_label in enumerate(classes):
                long_rows.append({"Model": name, "True_Label": true_label,
                                  "Predicted_Label": pred_label, "Count": int(cm[i, j])})

    fig.suptitle("Confusion Matrices — All Classification Models", fontweight="bold", fontsize=14)
    save_chart("01_confusion_matrices.png")

    cm_df = pd.DataFrame(long_rows)
    save_summary(cm_df, "confusion_matrix.csv")
    return cm_df


# ════════════════════════════════════════════════════════════════
# STEP 8 — ROC / PRECISION-RECALL CURVES
# ════════════════════════════════════════════════════════════════

def step8_roc_pr_curves(clf_models: dict, X_test, y_test) -> pd.DataFrame:
    print(section("STEP 8: ROC Curve, Precision-Recall Curve & ROC-AUC Scores"))

    classes = sorted(y_test.unique().tolist())
    y_test_bin = label_binarize(y_test, classes=classes)
    auc_rows = []
    colors = sns.color_palette(PALETTE_CAT, len(clf_models))

    fig, ax = plt.subplots(figsize=(9, 7))
    for color, (name, model) in zip(colors, clf_models.items()):
        if not hasattr(model, "predict_proba"):
            continue
        try:
            y_proba = model.predict_proba(X_test)
            fpr_all, tpr_all = [], []
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                fpr_all.append(fpr)
                tpr_all.append(tpr)
                auc_rows.append({"Model": name, "Class": classes[i], "AUC": round(auc(fpr, tpr), 4)})
            all_fpr = np.unique(np.concatenate(fpr_all))
            mean_tpr = np.zeros_like(all_fpr)
            for fpr, tpr in zip(fpr_all, tpr_all):
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            mean_tpr /= len(classes)
            macro_auc = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, color=color, linewidth=2, label=f"{name} (macro AUC={macro_auc:.3f})")
        except Exception as exc:
            print(f"    NOTE: ROC curve unavailable for '{name}': {exc}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_title("ROC Curve — Macro-Average (One-vs-Rest)", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    save_chart("02_roc_curve.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    for color, (name, model) in zip(colors, clf_models.items()):
        if not hasattr(model, "predict_proba"):
            continue
        try:
            y_proba = model.predict_proba(X_test)
            common_recall = np.linspace(0, 1, 100)
            mean_precision = np.zeros_like(common_recall)
            for i in range(len(classes)):
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
                mean_precision += np.interp(common_recall, recall[::-1], precision[::-1])
            mean_precision /= len(classes)
            ax.plot(common_recall, mean_precision, color=color, linewidth=2, label=name)
        except Exception as exc:
            print(f"    NOTE: PR curve unavailable for '{name}': {exc}")

    ax.set_title("Precision-Recall Curve — Macro-Average", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    save_chart("03_precision_recall_curve.png")

    auc_df = pd.DataFrame(auc_rows)
    save_summary(auc_df, "roc_auc_scores.csv")
    return auc_df


# ════════════════════════════════════════════════════════════════
# STEP 9 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════

def step9_feature_importance(clf_models: dict, reg_models: dict, feature_names) -> pd.DataFrame:
    print(section("STEP 9: Feature Importance"))

    rows = []
    for name, model in clf_models.items():
        imp = model.feature_importances_ if hasattr(model, "feature_importances_") else (
            np.abs(model.coef_).mean(axis=0) if hasattr(model, "coef_") else None)
        if imp is None:
            continue
        for feat, val in zip(feature_names, imp):
            rows.append({"Task": "Classification", "Model": name, "Feature": feat, "Importance": round(float(val), 6)})

    for name, model in reg_models.items():
        imp = model.feature_importances_ if hasattr(model, "feature_importances_") else (
            np.abs(model.coef_) if hasattr(model, "coef_") else None)
        if imp is None:
            continue
        for feat, val in zip(feature_names, imp):
            rows.append({"Task": "Regression", "Model": name, "Feature": feat, "Importance": round(float(val), 6)})

    imp_df = pd.DataFrame(rows)
    if imp_df.empty:
        print("  No models expose feature_importances_/coef_ — skipping.")
        return imp_df

    imp_df["Rank"] = imp_df.groupby(["Task", "Model"])["Importance"].rank(ascending=False, method="min").astype(int)
    imp_df = imp_df.sort_values(["Task", "Model", "Rank"]).reset_index(drop=True)
    save_summary(imp_df, "feature_importance.csv")

    for task in imp_df["Task"].unique():
        task_df = imp_df[imp_df["Task"] == task]
        avg_imp = task_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).head(20).reset_index()
        if avg_imp.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.barplot(data=avg_imp, y="Feature", x="Importance", palette="viridis", ax=ax)
        ax.set_title(f"Top 20 Most Important Features — {task}", fontweight="bold")
        ax.set_xlabel("Average Importance (across models)")
        save_chart(f"04_feature_importance_{task.lower()}.png")
        top_feat = avg_imp.iloc[0]
        add_insight(f"Feature with Highest Importance ({task}): '{top_feat['Feature']}' "
                   f"(avg importance {top_feat['Importance']:.4f})")

    return imp_df


# ════════════════════════════════════════════════════════════════
# STEP 10 — LEARNING & VALIDATION CURVES (best classifier + regressor)
# ════════════════════════════════════════════════════════════════

def step10_learning_validation_curves(clf_models, reg_models, best_clf_name, best_reg_name,
                                       X_full, y_class_full, y_reg_full) -> None:
    print(section("STEP 10: Learning Curve & Validation Curve"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            clf_models[best_clf_name], X_full, y_class_full, cv=3, scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 6), random_state=RANDOM_STATE,
        )
        axes[0].plot(train_sizes, train_scores.mean(axis=1), "o-", color=ACCENT_COLOR, label="Training Score")
        axes[0].plot(train_sizes, val_scores.mean(axis=1), "o-", color=RED_COLOR, label="Validation Score")
        axes[0].set_title(f"Learning Curve — {best_clf_name} (Classification, F1)", fontweight="bold")
        axes[0].set_xlabel("Training Set Size")
        axes[0].set_ylabel("F1 Score (weighted)")
        axes[0].legend()
    except Exception as exc:
        print(f"  WARNING: learning curve failed for classifier: {exc}")

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            reg_models[best_reg_name], X_full, y_reg_full, cv=3, scoring="r2",
            train_sizes=np.linspace(0.1, 1.0, 6), random_state=RANDOM_STATE,
        )
        axes[1].plot(train_sizes, train_scores.mean(axis=1), "o-", color=GREEN_COLOR, label="Training Score")
        axes[1].plot(train_sizes, val_scores.mean(axis=1), "o-", color=ORANGE_COLOR, label="Validation Score")
        axes[1].set_title(f"Learning Curve — {best_reg_name} (Regression, R2)", fontweight="bold")
        axes[1].set_xlabel("Training Set Size")
        axes[1].set_ylabel("R2 Score")
        axes[1].legend()
    except Exception as exc:
        print(f"  WARNING: learning curve failed for regressor: {exc}")
    save_chart("05_learning_curves.png")

    try:
        param_range = [2, 4, 6, 8, 10, 12, 15]
        train_scores, val_scores = validation_curve(
            clf_models[best_clf_name], X_full, y_class_full, param_name="max_depth",
            param_range=param_range, cv=3, scoring="f1_weighted", n_jobs=-1,
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(param_range, train_scores.mean(axis=1), "o-", color=ACCENT_COLOR, label="Training Score")
        ax.plot(param_range, val_scores.mean(axis=1), "o-", color=RED_COLOR, label="Validation Score")
        ax.set_title(f"Validation Curve — {best_clf_name} (max_depth)", fontweight="bold")
        ax.set_xlabel("max_depth")
        ax.set_ylabel("F1 Score (weighted)")
        ax.legend()
        save_chart("06_validation_curve.png")
    except Exception as exc:
        print(f"  NOTE: validation curve skipped ('{best_clf_name}' has no max_depth or failed): {exc}")


# ════════════════════════════════════════════════════════════════
# STEP 11 — CV SCORES BOXPLOT
# ════════════════════════════════════════════════════════════════

def step11_cv_scores_chart(clf_models: dict, reg_models: dict) -> None:
    print(section("STEP 11: Cross Validation Scores Chart"))

    for task, models in [("Classification", clf_models), ("Regression", reg_models)]:
        fold_rows = []
        for name in models.keys():
            scores = CV_FOLD_SCORES_CACHE.get((task, name))
            if scores is None:
                continue
            for i, s in enumerate(scores, 1):
                fold_rows.append({"Model": name, "Fold": i, "Score": s})
        if not fold_rows:
            continue
        fold_df = pd.DataFrame(fold_rows)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(data=fold_df, x="Model", y="Score", palette=PALETTE_CAT, ax=ax)
        sns.stripplot(data=fold_df, x="Model", y="Score", color="black", alpha=0.5, ax=ax)
        ax.set_title(f"Cross Validation Scores — {task}", fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        save_chart(f"07_cv_scores_{task.lower()}.png")


# ════════════════════════════════════════════════════════════════
# STEP 12 — REGRESSION DIAGNOSTIC PLOTS
# ════════════════════════════════════════════════════════════════

def step12_regression_diagnostics(reg_models: dict, X_test, y_test) -> None:
    print(section("STEP 12: Residual Plot & Prediction vs Actual"))

    n_models = len(reg_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, reg_models.items()):
        y_pred = model.predict(X_test)
        residuals = y_test.values - y_pred
        ax.scatter(y_pred, residuals, alpha=0.35, s=18, color=ACCENT_COLOR)
        ax.axhline(0, color=RED_COLOR, linestyle="--", linewidth=1.5)
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Residual")
    fig.suptitle("Residual Plots — All Regression Models", fontweight="bold", fontsize=14)
    save_chart("08_residual_plots.png")

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, reg_models.items()):
        y_pred = model.predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.35, s=18, color=GREEN_COLOR)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, color=RED_COLOR, linestyle="--", linewidth=1.5, label="Perfect Prediction")
        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.legend(fontsize=8)
    fig.suptitle("Prediction vs Actual — All Regression Models", fontweight="bold", fontsize=14)
    save_chart("09_prediction_vs_actual.png")


# ════════════════════════════════════════════════════════════════
# STEP 13 — CLASSIFICATION COMPARISON CHARTS
# ════════════════════════════════════════════════════════════════

def step13_classification_comparison_charts(clf_metrics: pd.DataFrame) -> None:
    print(section("STEP 13: Accuracy / Precision / Recall / F1 Comparison Charts"))

    if clf_metrics.empty:
        return
    chart_specs = [
        ("Accuracy", "10_accuracy_comparison.png", "Blues_r"),
        ("Precision", "11_precision_comparison.png", "Greens_r"),
        ("Recall", "12_recall_comparison.png", "Oranges_r"),
        ("F1_Score", "13_f1_score_comparison.png", "Purples_r"),
    ]
    for metric, filename, palette in chart_specs:
        ordered = clf_metrics.sort_values(metric, ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(data=ordered, x="Model", y=metric, palette=palette, ax=ax)
        ax.set_title(f"Model {metric.replace('_', ' ')} Comparison", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15)
        for p in bars.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                       ha="center", va="bottom", fontsize=9)
        save_chart(filename)


# ════════════════════════════════════════════════════════════════
# STEP 14 — MODEL RANKING (classification + regression, separately)
# ════════════════════════════════════════════════════════════════

def step14_model_ranking(clf_metrics: pd.DataFrame, cv_df: pd.DataFrame, reg_metrics: pd.DataFrame):
    print(section("STEP 14: Model Ranking"))

    clf_rank = clf_metrics.copy()
    if not cv_df.empty:
        clf_cv = cv_df[cv_df["Task"] == "Classification"][["Model", "Mean_CV_Score", "Std_Dev"]]
        clf_rank = clf_rank.merge(clf_cv, on="Model", how="left")
    rank_cols = []
    for metric in ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "Mean_CV_Score"]:
        if metric in clf_rank.columns:
            col = f"{metric}_Rank"
            clf_rank[col] = clf_rank[metric].rank(ascending=False, method="min")
            rank_cols.append(col)
    clf_rank["Overall_Rank"] = clf_rank[rank_cols].mean(axis=1).rank(ascending=True, method="min").astype(int)
    clf_rank = clf_rank.drop(columns=rank_cols).sort_values("Overall_Rank").reset_index(drop=True)
    save_summary(clf_rank, "classification_model_ranking.csv")

    reg_rank = reg_metrics.copy()
    if not cv_df.empty:
        reg_cv = cv_df[cv_df["Task"] == "Regression"][["Model", "Mean_CV_Score"]]
        reg_rank = reg_rank.merge(reg_cv, on="Model", how="left")
    reg_rank["R2_Rank"] = reg_rank["R2_Score"].rank(ascending=False, method="min")
    reg_rank["RMSE_Rank"] = reg_rank["RMSE"].rank(ascending=True, method="min")
    reg_rank["MAE_Rank"] = reg_rank["MAE"].rank(ascending=True, method="min")
    rank_cols_r = ["R2_Rank", "RMSE_Rank", "MAE_Rank"]
    if "Mean_CV_Score" in reg_rank.columns:
        reg_rank["CV_Rank"] = reg_rank["Mean_CV_Score"].rank(ascending=False, method="min")
        rank_cols_r.append("CV_Rank")
    reg_rank["Overall_Rank"] = reg_rank[rank_cols_r].mean(axis=1).rank(ascending=True, method="min").astype(int)
    reg_rank = reg_rank.drop(columns=rank_cols_r).sort_values("Overall_Rank").reset_index(drop=True)
    save_summary(reg_rank, "regression_model_ranking.csv")

    print("Classification Ranking:\n", clf_rank[["Model", "Overall_Rank"]].to_string(index=False))
    print("\nRegression Ranking:\n", reg_rank[["Model", "Overall_Rank"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, rank_df, title, metric_col, color in [
        (axes[0], clf_rank, "Classification Model Ranking", "F1_Score", "Blues_r"),
        (axes[1], reg_rank, "Regression Model Ranking", "R2_Score", "Greens_r"),
    ]:
        ordered = rank_df.sort_values("Overall_Rank")
        ax.barh(ordered["Model"], ordered[metric_col], color=sns.color_palette(color, len(ordered)))
        ax.invert_yaxis()
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(metric_col)
    save_chart("14_model_ranking.png")

    return clf_rank, reg_rank


# ════════════════════════════════════════════════════════════════
# STEP 15 — STATISTICAL EVALUATION (paired t-test, top-2 classifiers)
# ════════════════════════════════════════════════════════════════

def step15_statistical_evaluation(clf_models: dict, X_full, y_class_full, ranking: pd.DataFrame) -> str:
    print(section("STEP 15: Statistical Evaluation"))

    if ranking.empty or len(ranking) < 2:
        return "Not enough models for a paired statistical comparison."

    top2 = ranking.sort_values("Overall_Rank").head(2)["Model"].tolist()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    try:
        scores_a = cross_val_score(clf_models[top2[0]], X_full, y_class_full, cv=skf, scoring="f1_weighted", n_jobs=-1)
        scores_b = cross_val_score(clf_models[top2[1]], X_full, y_class_full, cv=skf, scoring="f1_weighted", n_jobs=-1)
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        significant = "statistically significant" if p_value < 0.05 else "not statistically significant"
        text = (
            f"Paired t-test comparing the top 2 ranked models on {CV_FOLDS}-fold CV F1 scores:\n"
            f"  {top2[0]} (mean F1={scores_a.mean():.4f}) vs {top2[1]} (mean F1={scores_b.mean():.4f})\n"
            f"  t-statistic={t_stat:.4f}, p-value={p_value:.4f} -> difference is {significant} (alpha=0.05)."
        )
        print(text)
        add_insight(f"Statistical comparison of top 2 models: difference is {significant} (p={p_value:.4f}).")
        return text
    except Exception as exc:
        return f"Statistical comparison could not be completed: {exc}"


# ════════════════════════════════════════════════════════════════
# STEP 16 — OVERFITTING / UNDERFITTING DETECTION
# ════════════════════════════════════════════════════════════════

def step16_overfit_underfit_detection(clf_metrics: pd.DataFrame, reg_metrics: pd.DataFrame,
                                      gap_threshold: float = 0.08) -> pd.DataFrame:
    print(section("STEP 16: Overfitting / Underfitting Detection"))

    rows = []
    for _, row in clf_metrics.iterrows():
        gap = row["Train_Accuracy"] - row["Accuracy"]
        status = "Overfitting" if gap > gap_threshold else ("Underfitting" if row["Accuracy"] < 0.55 else "Well Generalized")
        rows.append({"Model": row["Model"], "Task": "Classification",
                    "Train_Score": row["Train_Accuracy"], "Test_Score": row["Accuracy"],
                    "Gap": round(gap, 4), "Status": status})
    for _, row in reg_metrics.iterrows():
        gap = row["Train_R2"] - row["R2_Score"]
        status = "Overfitting" if gap > gap_threshold else ("Underfitting" if row["R2_Score"] < 0.3 else "Well Generalized")
        rows.append({"Model": row["Model"], "Task": "Regression",
                    "Train_Score": row["Train_R2"], "Test_Score": row["R2_Score"],
                    "Gap": round(gap, 4), "Status": status})

    fit_df = pd.DataFrame(rows)
    save_summary(fit_df, "overfitting_underfitting_detection.csv")
    print(fit_df.to_string(index=False))

    for _, row in fit_df.iterrows():
        if row["Status"] != "Well Generalized":
            add_insight(f"{row['Status']} detected — {row['Model']} ({row['Task']}): "
                       f"train={row['Train_Score']:.3f}, test={row['Test_Score']:.3f}, gap={row['Gap']:.3f}")

    return fit_df


# ════════════════════════════════════════════════════════════════
# STEP 17 — BEST MODEL SUMMARY & BUSINESS INSIGHTS
# ════════════════════════════════════════════════════════════════

def step17_best_model_summary(clf_rank: pd.DataFrame, reg_rank: pd.DataFrame,
                              fit_df: pd.DataFrame) -> pd.DataFrame:
    print(section("STEP 17: Best Model Summary & Business Insights"))

    rows = []
    best_c = clf_rank.sort_values("Overall_Rank").iloc[0]
    worst_c = clf_rank.sort_values("Overall_Rank").iloc[-1]
    rows.append({"Task": "Classification", "Best_Model": best_c["Model"],
                "Key_Metric_1": f"Accuracy={best_c['Accuracy']:.4f}", "Key_Metric_2": f"F1={best_c['F1_Score']:.4f}",
                "Key_Metric_3": f"ROC_AUC={best_c.get('ROC_AUC', np.nan):.4f}"})
    add_insight(f"Best Performing Classification Model: {best_c['Model']} (F1={best_c['F1_Score']:.4f})")
    add_insight(f"Worst Performing Classification Model: {worst_c['Model']} (F1={worst_c['F1_Score']:.4f})")

    best_r = reg_rank.sort_values("Overall_Rank").iloc[0]
    rows.append({"Task": "Regression", "Best_Model": best_r["Model"],
                "Key_Metric_1": f"R2={best_r['R2_Score']:.4f}", "Key_Metric_2": f"RMSE={best_r['RMSE']:.4f}",
                "Key_Metric_3": f"MAE={best_r['MAE']:.4f}"})
    add_insight(f"Best Performing Regression Model: {best_r['Model']} (R2={best_r['R2_Score']:.4f})")

    if not fit_df.empty:
        well_gen = fit_df[fit_df["Status"] == "Well Generalized"].sort_values("Gap")
        if not well_gen.empty:
            bg = well_gen.iloc[0]
            add_insight(f"Best Generalization Model (smallest train-test gap): {bg['Model']} ({bg['Task']}), gap={bg['Gap']:.4f}")

    summary_df = pd.DataFrame(rows)
    save_summary(summary_df, "best_model_summary.csv")
    print(summary_df.to_string(index=False))
    return summary_df


# ════════════════════════════════════════════════════════════════
# STEP 18 — RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════

def step18_recommendation_engine(clf_rank: pd.DataFrame, reg_rank: pd.DataFrame,
                                 fit_df: pd.DataFrame, imp_df: pd.DataFrame) -> str:
    print(section("STEP 18: Recommendation Engine"))

    recs = []
    best_c = clf_rank.sort_values("Overall_Rank").iloc[0]
    best_r = reg_rank.sort_values("Overall_Rank").iloc[0]
    recs.append(
        "1. BEST MODEL FOR PRODUCTION\n"
        f"   For Delay Status classification, '{best_c['Model']}' ranks #1 overall "
        f"(F1={best_c['F1_Score']:.3f}). For Shipping Lead Time regression, '{best_r['Model']}' "
        f"ranks #1 overall (R2={best_r['R2_Score']:.3f}). Recommend staging + shadow-mode "
        "monitoring before full production rollout."
    )

    overfit_models = fit_df[fit_df["Status"] == "Overfitting"]["Model"].tolist() if not fit_df.empty else []
    recs.append(
        "2. HYPERPARAMETER TUNING SUGGESTIONS\n"
        + (f"   {', '.join(overfit_models)} show overfitting — reduce max_depth / n_estimators, "
           "add regularization. " if overfit_models else
           "   No strong overfitting detected; a grid/random search over depth, n_estimators, "
           "and learning rate is still recommended. ")
        + "Use the same Stratified K-Fold / K-Fold scheme used in this evaluation."
    )

    if not imp_df.empty:
        top_feat = imp_df[imp_df["Task"] == "Classification"].groupby("Feature")["Importance"].mean().sort_values(ascending=False)
        top_feat_name = top_feat.index[0] if not top_feat.empty else "N/A"
        recs.append(
            "3. FEATURE ENGINEERING IMPROVEMENTS\n"
            f"   '{top_feat_name}' is currently the most influential feature. Consider adding "
            "distance/geodesic proxy (Factory to customer State), carrier on-time history, "
            "and interaction features (Ship Mode x Region)."
        )

    recs.append(
        "4. DATA COLLECTION IMPROVEMENTS\n"
        "   Both targets in this model are derived from a SIMULATED shipping lead time (see "
        "01_data_cleaning.py: the real Ship Date column is corrupted for 100% of records). "
        "Prioritize capturing real, validated ship-scan timestamps so future models learn from "
        "observed outcomes, not simulated ones."
    )

    recs.append(
        "5. HANDLING CLASS IMBALANCE\n"
        "   If 'Delayed' / 'Moderate Delay' / 'On Time' classes are unevenly distributed, apply "
        "class_weight='balanced', or prefer F1/ROC-AUC (already used here) over raw Accuracy."
    )

    recs.append(
        "6. MODEL DEPLOYMENT READINESS\n"
        "   Before production: wrap preprocessing + model in a single sklearn Pipeline, version "
        "model artifacts with metadata.json, and set up periodic re-evaluation against this same "
        "script as new shipment data arrives."
    )

    report_text = (
        "MODEL EVALUATION — BUSINESS & TECHNICAL RECOMMENDATIONS\n"
        "Factory-to-Customer Shipping Route Efficiency Analysis\nNassau Candy Distributor\n"
        + "=" * 70 + "\n\n" + "\n\n".join(recs) + "\n"
    )
    for rec in recs:
        print(f"\n  {rec}")
    return report_text


# ════════════════════════════════════════════════════════════════
# STEP 19 — FINAL EXECUTIVE REPORT
# ════════════════════════════════════════════════════════════════

def step19_final_report(df, clf_metrics, reg_metrics, cv_df, clf_rank, reg_rank,
                        best_summary, stats_text, recommendations_text) -> None:
    print(section("STEP 19: Final Executive Report"))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 70, "EXECUTIVE REPORT — MACHINE LEARNING MODEL EVALUATION",
        "Factory-to-Customer Shipping Route Efficiency Analysis", "Nassau Candy Distributor",
        f"Generated: {ts}", "=" * 70, "",
        section("EXECUTIVE SUMMARY").strip(),
        ("This report evaluates a classification suite predicting Delay Status (On Time / "
         "Moderate Delay / Delayed) and a regression suite predicting Shipping Lead Time "
         "(Simulated) for the Nassau Candy shipping analytics project. Models are scored on "
         "held-out test data, validated with cross-validation, and ranked on a composite score."),
        "", section("DATASET INFORMATION").strip(),
        f"  Source file            : {DATA_PATH}",
        f"  Total rows evaluated   : {len(df):,}",
        f"  Classification target  : {DELAY_COL}",
        f"  Regression target      : {LEAD_TIME_COL}",
        f"  Train/Test split       : {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        f"  Cross-validation folds : {CV_FOLDS}",
        "", section("PERFORMANCE COMPARISON — CLASSIFICATION").strip(),
        clf_metrics.to_string(index=False) if not clf_metrics.empty else "  No results.",
        "", section("PERFORMANCE COMPARISON — REGRESSION").strip(),
        reg_metrics.to_string(index=False) if not reg_metrics.empty else "  No results.",
        "", section("STATISTICAL EVALUATION").strip(), stats_text,
        "", section("BUSINESS & MODEL INSIGHTS").strip(),
    ]
    for insight in insight_list:
        lines.append(f"  - {insight}")

    lines += ["", section("BEST MODEL SELECTION").strip()]
    for _, row in best_summary.iterrows():
        lines.append(f"  [{row['Task']}] Best Model: {row['Best_Model']} | {row['Key_Metric_1']} | "
                    f"{row['Key_Metric_2']} | {row['Key_Metric_3']}")

    lines += ["", section("RECOMMENDATIONS").strip(),
             recommendations_text.split("=" * 70, 1)[-1].strip(),
             "", section("CONCLUSION").strip(),
             textwrap.fill(
                 "The top-ranked classification and regression models provide a solid, explainable "
                 "baseline for production deployment. Because the underlying lead-time labels are "
                 "currently simulated (see the 01_data_cleaning.py headline data-quality finding), "
                 "model performance should be re-validated once genuine Ship Date data becomes "
                 "available.", width=72),
             "", "=" * 70, "END OF REPORT", "=" * 70]

    save_report_text("\n".join(lines) + "\n", "model_evaluation_report.txt")


# ════════════════════════════════════════════════════════════════
# STEP 20 — FINAL EXECUTION SUMMARY
# ════════════════════════════════════════════════════════════════

def step20_execution_summary() -> None:
    print(section("STEP 20: Final Execution Summary"))
    print(f"  Total Charts Generated      : {chart_count}")
    print(f"  Total Summary Tables Saved  : {summary_count}")
    print(f"  Total Text Reports Saved    : {report_count}")
    print(f"  Total Insights Generated    : {len(insight_list)}")
    print(f"  Charts saved in             : ./{CHARTS_DIR}/")
    print(f"  Summaries & reports saved in: ./{REPORTS_DIR}/")
    print(f"  Models saved in             : ./{MODELS_DIR}/")
    print("  Model Training & Evaluation Completed Successfully")
    print("=" * 70)


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main() -> None:
    try:
        df = step1_load_dataset()
        X, y_class, y_reg = step2_build_feature_matrix(df)
        (clf_models, reg_models, X_train, X_test,
         y_train_c, y_test_c, y_train_r, y_test_r) = step3_train_models(X, y_class, y_reg)

        clf_metrics = step4_classification_metrics(clf_models, X_train, X_test, y_train_c, y_test_c)
        reg_metrics = step5_regression_metrics(reg_models, X_train, X_test, y_train_r, y_test_r)

        X_full = pd.concat([X_train, X_test], axis=0)
        y_class_full = pd.concat([y_train_c, y_test_c], axis=0)
        y_reg_full = pd.concat([y_train_r, y_test_r], axis=0)
        cv_df = step6_cross_validation(clf_models, reg_models, X_full, y_class_full, y_reg_full)

        step7_confusion_matrices(clf_models, X_test, y_test_c)
        step8_roc_pr_curves(clf_models, X_test, y_test_c)
        imp_df = step9_feature_importance(clf_models, reg_models, X.columns.tolist())

        best_clf_name = clf_metrics.sort_values("F1_Score", ascending=False).iloc[0]["Model"]
        best_reg_name = reg_metrics.sort_values("R2_Score", ascending=False).iloc[0]["Model"]
        step10_learning_validation_curves(clf_models, reg_models, best_clf_name, best_reg_name,
                                          X_full, y_class_full, y_reg_full)
        step11_cv_scores_chart(clf_models, reg_models)
        step12_regression_diagnostics(reg_models, X_test, y_test_r)
        step13_classification_comparison_charts(clf_metrics)

        clf_rank, reg_rank = step14_model_ranking(clf_metrics, cv_df, reg_metrics)
        stats_text = step15_statistical_evaluation(clf_models, X_full, y_class_full, clf_rank)
        fit_df = step16_overfit_underfit_detection(clf_metrics, reg_metrics)
        best_summary = step17_best_model_summary(clf_rank, reg_rank, fit_df)
        recs_text = step18_recommendation_engine(clf_rank, reg_rank, fit_df, imp_df)
        step19_final_report(df, clf_metrics, reg_metrics, cv_df, clf_rank, reg_rank,
                            best_summary, stats_text, recs_text)
        step20_execution_summary()

    except Exception as exc:
        print(f"\nPIPELINE FAILED: {exc}")
        raise


if __name__ == "__main__":
    main()