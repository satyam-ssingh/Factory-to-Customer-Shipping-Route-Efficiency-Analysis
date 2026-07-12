"""
06_region_analysis.py
--------------------------
Phase 6 - Region-Level Shipping Performance Analysis
Project: Factory-to-Customer Shipping Route Efficiency Analysis
         for Nassau Candy Distributor

Input  : featured_nassau_candy.csv  (output of 02_feature_engineering.py)
Output : EDA_Charts/      (region-level visualizations)
         EDA_Summaries/   (region summary tables + reports)

Run after:
  01_data_cleaning.py
  02_feature_engineering.py
  03_exploratory_data_analysis.py
  04_ship_mode_analysis.py
  05_state_analysis.py
"""

import os
import textwrap
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "DejaVu Sans",
})

PALETTE_CAT  = "Set2"
PALETTE_SEQ  = "Blues_r"
ACCENT_COLOR = "#2C7BB6"
GREEN_COLOR  = "#27AE60"
RED_COLOR    = "#E74C3C"
ORANGE_COLOR = "#E67E22"

DELAY_STATUS_COLORS = {
    "On Time":        "#2ECC71",
    "Moderate Delay": "#F1C40F",
    "Delayed":        "#E74C3C",
}
EFF_SCORE_ORDER = ["Excellent", "Good", "Average", "Poor"]

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
IN_PATH       = "featured_nassau_candy.csv"
CHARTS_DIR    = "EDA_Charts"
SUMMARIES_DIR = "EDA_Summaries"
os.makedirs(CHARTS_DIR,    exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

LEAD_TIME_COL = "Shipping Lead Time (Simulated)"
REGION_COL    = "Region"

# ─────────────────────────────────────────────────────────────
# COUNTERS & GLOBAL ACCUMULATORS
# ─────────────────────────────────────────────────────────────
chart_count   = 0
summary_count = 0
report_count  = 0
insight_list  = []


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def save_chart(filename: str) -> None:
    """Save the current matplotlib figure to EDA_Charts/ and close it."""
    global chart_count
    path = os.path.join(CHARTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    chart_count += 1
    print(f"  [Chart saved]   {path}")


def save_summary(data: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to EDA_Summaries/ as a CSV file."""
    global summary_count
    path = os.path.join(SUMMARIES_DIR, filename)
    data.to_csv(path, index=False, float_format="%.2f")
    summary_count += 1
    print(f"  [Summary saved] {path}")


def save_report_text(text: str, filename: str) -> None:
    """Save a plain-text report to EDA_Summaries/."""
    global report_count
    path = os.path.join(SUMMARIES_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        report_count += 1
        print(f"  [Report saved]  {path}")
    except OSError as exc:
        print(f"  WARNING: Could not save report '{filename}': {exc}")


def add_insight(text: str) -> None:
    """Append a business insight to the global list and print it."""
    insight_list.append(text)
    print(f"  [Insight {len(insight_list):02d}] {text}")


def fmt_currency(val: float) -> str:
    """Format a number as a dollar string, e.g. $1,234.56."""
    return f"${val:,.2f}"


def fmt_pct(val: float) -> str:
    """Format a number as a percentage string, e.g. 12.34%."""
    return f"{val:.2f}%"


def top_n_regions(data: pd.DataFrame, col: str, n: int = 10,
                   ascending: bool = False) -> pd.DataFrame:
    """Return the top-N (or bottom-N) region rows ranked by *col*."""
    return (
        data.sort_values(col, ascending=ascending)
            .head(n)
            .reset_index(drop=True)
    )


def section(title: str) -> str:
    """Build a formatted section header string for text reports."""
    border = "=" * 60
    return f"\n{border}\n{title}\n{border}\n"


# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATASET
# ════════════════════════════════════════════════════════════════

def step1_load_dataset(path: str = IN_PATH) -> pd.DataFrame:
    """
    Load the featured dataset produced by 02_feature_engineering.py,
    validate required columns and print a structural overview.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Load Dataset")
    print("=" * 60)

    try:
        data = pd.read_csv(path, low_memory=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"'{path}' not found. "
            "Run 01_data_cleaning.py and 02_feature_engineering.py first."
        ) from exc

    required = [
        REGION_COL, "Sales", "Gross Profit", "Profit Margin %",
        LEAD_TIME_COL, "Delay Status", "Route Efficiency Score",
        "Order ID", "State/Province", "Factory",
    ]
    missing_req = [c for c in required if c not in data.columns]
    if missing_req:
        raise KeyError(
            f"Required column(s) missing: {missing_req}. "
            "Re-run 02_feature_engineering.py to regenerate the featured dataset."
        )

    print(f"Dataset Shape       : {data.shape}")
    print(f"Total Rows          : {data.shape[0]:,}")
    print(f"Total Columns       : {data.shape[1]}")
    print(f"Unique Regions      : {data[REGION_COL].nunique()}")
    print(f"Unique States       : {data['State/Province'].nunique()}")
    print(f"Date Range          : {data['Order Date'].min()} — {data['Order Date'].max()}")

    return data


# ════════════════════════════════════════════════════════════════
# STEP 2 — DATA QUALITY CHECK
# ════════════════════════════════════════════════════════════════

def step2_data_quality(data: pd.DataFrame) -> None:
    """
    Check for missing values, duplicate rows and confirm that the
    key region-analysis columns are complete.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Data Quality Check")
    print("=" * 60)

    ignore_cols = ["Ship Date", "Shipping Lead Time"]
    check_df = data.drop(columns=ignore_cols, errors="ignore")

    missing = check_df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  Missing Values  : None — dataset is complete.")
    else:
        miss_pct = (missing / len(data) * 100).round(2)
        print("  Missing Values:")
        print(pd.concat([missing.rename("Count"), miss_pct.rename("%")],
                        axis=1).to_string())

    dup_count = data.duplicated().sum()
    print(f"\n  Duplicate Rows  : {dup_count}")

    print(f"\n  Regions found   : {sorted(data[REGION_COL].dropna().unique().tolist())}")
    us_count = data.get("Is US Record", pd.Series([True] * len(data))).sum()
    ca_count = len(data) - us_count
    print(f"  US Records      : {int(us_count):,}")
    print(f"  Canadian Records: {int(ca_count):,}")


# ════════════════════════════════════════════════════════════════
# STEP 3 — BUILD CORE REGION SUMMARY
# ════════════════════════════════════════════════════════════════

def step3_build_region_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate key KPIs per region:
      Total Shipments, Total Sales, Total Gross Profit, Total Units,
      Total Cost, Average Lead Time, Average Profit Margin,
      On-Time Rate %, Moderate Delay %, Delay Rate %.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Core Region Summary")
    print("=" * 60)

    summary = (
        data.groupby(REGION_COL)
        .agg(
            Total_Shipments=("Order ID",     "count"),
            Total_Sales=    ("Sales",        "sum"),
            Total_Profit=   ("Gross Profit", "sum"),
            Total_Units=    ("Units",        "sum"),
            Total_Cost=     ("Cost",         "sum"),
            Avg_Lead_Time=  (LEAD_TIME_COL,  "mean"),
            Avg_Profit_Margin=("Profit Margin %", "mean"),
        )
        .round(2)
        .reset_index()
    )

    # ── Delay Status breakdown per region ─────────────────────
    if "Delay Status" in data.columns:
        status_counts = (
            data.groupby([REGION_COL, "Delay Status"])
            .size()
            .unstack(fill_value=0)
        )
        status_pct = (
            status_counts
            .div(status_counts.sum(axis=1), axis=0)
            .mul(100)
            .round(2)
        )
        for col in ["On Time", "Moderate Delay", "Delayed"]:
            if col not in status_pct.columns:
                status_pct[col] = 0.0
        status_pct = status_pct.rename(columns={
            "On Time":        "On_Time_Rate_%",
            "Moderate Delay": "Moderate_Delay_%",
            "Delayed":        "Delay_Rate_%",
        }).reset_index()
        summary = summary.merge(
            status_pct[[REGION_COL, "On_Time_Rate_%",
                         "Moderate_Delay_%", "Delay_Rate_%"]],
            on=REGION_COL, how="left",
        )
    else:
        summary["On_Time_Rate_%"]   = np.nan
        summary["Moderate_Delay_%"] = np.nan
        summary["Delay_Rate_%"]     = np.nan

    # ── Dominant Factory serving each region ──────────────────
    if "Factory" in data.columns:
        dom_factory = (
            data.groupby(REGION_COL)["Factory"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
            .rename("Dominant_Factory")
            .reset_index()
        )
        summary = summary.merge(dom_factory, on=REGION_COL, how="left")

    # ── Number of states served per region ─────────────────────
    if "State/Province" in data.columns:
        state_count = (
            data.groupby(REGION_COL)["State/Province"]
            .nunique()
            .rename("States_Served")
            .reset_index()
        )
        summary = summary.merge(state_count, on=REGION_COL, how="left")

    summary = summary.fillna(0)
    summary = summary.sort_values("Total_Sales", ascending=False).reset_index(drop=True)

    save_summary(summary, "region_summary.csv")
    print(f"  Regions captured : {len(summary)}")
    print(summary.to_string(index=False))

    return summary


# ════════════════════════════════════════════════════════════════
# STEP 4 — ROUTE EFFICIENCY BY REGION
# ════════════════════════════════════════════════════════════════

def step4_route_efficiency_by_region(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Route Efficiency Score percentage distribution
    (Excellent / Good / Average / Poor) for each region.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Route Efficiency by Region")
    print("=" * 60)

    if "Route Efficiency Score" not in data.columns:
        print("  Route Efficiency Score column not found — skipping.")
        return pd.DataFrame()

    counts = (
        data.groupby([REGION_COL, "Route Efficiency Score"])
        .size()
        .unstack(fill_value=0)
    )
    pct = (
        counts
        .div(counts.sum(axis=1), axis=0)
        .mul(100)
        .round(2)
    )
    for col in EFF_SCORE_ORDER:
        if col not in pct.columns:
            pct[col] = 0.0
    pct = pct[EFF_SCORE_ORDER].reset_index()

    save_summary(pct, "region_route_efficiency.csv")
    print(pct.to_string(index=False))

    return pct


# ════════════════════════════════════════════════════════════════
# STEP 5 — REGION COMPARISON TABLE
# ════════════════════════════════════════════════════════════════

def step5_region_comparison(summary: pd.DataFrame,
                             efficiency: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the core region summary with Route Efficiency Score
    distribution into one consolidated comparison table.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Region Comparison Table")
    print("=" * 60)

    comparison = summary.copy()
    if not efficiency.empty:
        comparison = comparison.merge(efficiency, on=REGION_COL, how="left")

    save_summary(comparison, "region_comparison.csv")
    print(f"  Comparison table shape: {comparison.shape}")
    print(comparison.to_string(index=False))

    return comparison


# ════════════════════════════════════════════════════════════════
# STEP 6 — DELAY SUMMARY BY REGION
# ════════════════════════════════════════════════════════════════

def step6_delay_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build a detailed delay summary per region: raw counts and
    percentage for each Delay Status category.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Delay Summary by Region")
    print("=" * 60)

    if "Delay Status" not in data.columns:
        print("  Delay Status column not found — skipping.")
        return pd.DataFrame()

    counts = (
        data.groupby([REGION_COL, "Delay Status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    totals = counts.set_index(REGION_COL).sum(axis=1).rename("Total")
    counts = counts.set_index(REGION_COL)
    for col in ["On Time", "Moderate Delay", "Delayed"]:
        if col in counts.columns:
            counts[f"{col} %"] = (counts[col] / totals * 100).round(2)
    counts = counts.reset_index()
    counts = (
        counts.sort_values("Delayed", ascending=False)
        if "Delayed" in counts.columns else counts
    )
    counts = counts.reset_index(drop=True)

    save_summary(counts, "region_delay_summary.csv")
    print(counts.to_string(index=False))

    return counts


# ════════════════════════════════════════════════════════════════
# STEP 7 — STATISTICAL SUMMARY
# ════════════════════════════════════════════════════════════════

def step7_statistical_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR
    for Sales, Gross Profit and Shipping Lead Time, broken down by
    region.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Statistical Summary by Region")
    print("=" * 60)

    target_cols = ["Sales", "Gross Profit", LEAD_TIME_COL]
    target_cols = [c for c in target_cols if c in data.columns]

    stat_rows = []
    for region, grp in data.groupby(REGION_COL):
        for col in target_cols:
            series = pd.to_numeric(grp[col], errors="coerce").dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            stat_rows.append({
                REGION_COL: region,
                "Metric":   col,
                "Mean":     round(series.mean(),   2),
                "Median":   round(series.median(), 2),
                "Std Dev":  round(series.std(),    2),
                "Variance": round(series.var(),    2),
                "Min":      round(series.min(),    2),
                "Max":      round(series.max(),    2),
                "Q1":       round(q1, 2),
                "Q3":       round(q3, 2),
                "IQR":      round(iqr, 2),
            })

    stats_df = pd.DataFrame(stat_rows)
    save_summary(stats_df, "region_statistics.csv")
    print(f"  Statistical records generated: {len(stats_df):,}")
    print(stats_df.to_string(index=False))

    return stats_df


# ════════════════════════════════════════════════════════════════
# STEP 8 — OUTLIER DETECTION (IQR METHOD)
# ════════════════════════════════════════════════════════════════

def step8_outlier_detection(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in Sales, Gross Profit and Shipping Lead Time
    using the 1.5 x IQR method, reported per region.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Outlier Detection (IQR Method)")
    print("=" * 60)

    target_cols = ["Sales", "Gross Profit", LEAD_TIME_COL]
    target_cols = [c for c in target_cols if c in data.columns]

    outlier_rows = []
    for col in target_cols:
        series = pd.to_numeric(data[col], errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr   = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        is_out = (series < lower) | (series > upper)

        temp = data.loc[series.notna()].copy()
        temp["__out__"] = is_out.loc[series.notna()]

        for region, grp in temp.groupby(REGION_COL):
            total = len(grp)
            n_out = int(grp["__out__"].sum())
            outlier_rows.append({
                REGION_COL:      region,
                "Column":        col,
                "Lower Fence":   round(lower, 2),
                "Upper Fence":   round(upper, 2),
                "Total Records": total,
                "Outlier Count": n_out,
                "Outlier %":     round(n_out / total * 100, 2) if total else 0.0,
            })

    outliers_df = pd.DataFrame(outlier_rows)
    outliers_df = outliers_df.sort_values(
        ["Column", "Outlier %"], ascending=[True, False]
    ).reset_index(drop=True)

    save_summary(outliers_df, "region_outliers.csv")
    print(f"  Outlier records detected: {outliers_df['Outlier Count'].sum():,}")
    print(outliers_df.to_string(index=False))

    return outliers_df


# ════════════════════════════════════════════════════════════════
# STEP 9 — CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════

def step9_correlation_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and visualize a correlation matrix for the key numeric
    columns: Sales, Gross Profit, Profit Margin %, Lead Time, Units, Cost.
    """
    print("\n" + "=" * 60)
    print("STEP 9: Correlation Analysis")
    print("=" * 60)

    candidate_cols = ["Sales", "Gross Profit", "Profit Margin %",
                      LEAD_TIME_COL, "Units", "Cost"]
    available = [c for c in candidate_cols if c in data.columns]
    skipped   = [c for c in candidate_cols if c not in data.columns]
    if skipped:
        print(f"  Columns not found and skipped: {skipped}")

    if len(available) < 2:
        print("  WARNING: Not enough numeric columns for correlation analysis.")
        return pd.DataFrame()

    try:
        corr_df     = data[available].apply(pd.to_numeric, errors="coerce").dropna()
        corr_matrix = corr_df.corr().round(3)
    except Exception as exc:
        print(f"  WARNING: Correlation computation failed: {exc}")
        return pd.DataFrame()

    save_summary(
        corr_matrix.reset_index().rename(columns={"index": "Column"}),
        "region_correlation.csv",
    )
    print(corr_matrix.to_string())

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Pearson r"}, ax=ax,
    )
    ax.set_title("Correlation Heatmap — Key Region Metrics", fontweight="bold")
    save_chart("06_region_correlation_heatmap.png")

    return corr_matrix


# ════════════════════════════════════════════════════════════════
# STEP 10 — ANOVA TEST
# ════════════════════════════════════════════════════════════════

def step10_anova_test(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a one-way ANOVA to determine whether Region significantly
    affects Sales, Gross Profit and Shipping Lead Time.
    """
    print("\n" + "=" * 60)
    print("STEP 10: ANOVA Test (Region Effect)")
    print("=" * 60)

    target_cols = ["Sales", "Gross Profit", LEAD_TIME_COL]
    target_cols = [c for c in target_cols if c in data.columns]

    anova_rows = []
    for col in target_cols:
        try:
            groups = [
                pd.to_numeric(grp[col], errors="coerce").dropna()
                for _, grp in data.groupby(REGION_COL)
            ]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                continue
            f_stat, p_val = stats.f_oneway(*groups)
            anova_rows.append({
                "Metric":               col,
                "F-Statistic":          round(f_stat, 4),
                "P-Value":              round(p_val, 6),
                "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No",
            })
        except Exception as exc:
            print(f"  WARNING: ANOVA failed for '{col}': {exc}")

    anova_df = pd.DataFrame(anova_rows)

    if not anova_df.empty:
        save_summary(anova_df, "region_anova_results.csv")
        print(anova_df.to_string(index=False))
        for _, row in anova_df.iterrows():
            add_insight(
                f"ANOVA — {row['Metric']}: F={row['F-Statistic']}, "
                f"p={row['P-Value']} -> Region effect significant: "
                f"{row['Significant (p<0.05)']}"
            )
    else:
        print("  WARNING: No ANOVA results generated.")

    return anova_df


# ════════════════════════════════════════════════════════════════
# STEP 11 — REGION RANKING
# ════════════════════════════════════════════════════════════════

def step11_region_ranking(summary: pd.DataFrame,
                           efficiency: pd.DataFrame) -> pd.DataFrame:
    """
    Rank every region on Sales, Profit, Shipment Volume, Lead Time,
    Delay Rate, Route Efficiency and Profit Margin, then compute an
    Overall Rank from the average of the individual rank scores
    (lower = better).
    """
    print("\n" + "=" * 60)
    print("STEP 11: Region Ranking")
    print("=" * 60)

    ranking = summary[[REGION_COL]].copy()

    ranking["Sales Rank"] = (
        summary["Total_Sales"].rank(ascending=False, method="min").astype(int)
    )
    ranking["Profit Rank"] = (
        summary["Total_Profit"].rank(ascending=False, method="min").astype(int)
    )
    ranking["Shipment Rank"] = (
        summary["Total_Shipments"].rank(ascending=False, method="min").astype(int)
    )
    ranking["Lead Time Rank"] = (
        summary["Avg_Lead_Time"].rank(ascending=True, method="min").astype(int)
    )

    rank_cols = ["Sales Rank", "Profit Rank", "Shipment Rank", "Lead Time Rank"]

    if "Delay_Rate_%" in summary.columns:
        ranking["Delay Rank"] = (
            summary["Delay_Rate_%"].rank(ascending=True, method="min").astype(int)
        )
        rank_cols.append("Delay Rank")

    if "Avg_Profit_Margin" in summary.columns:
        ranking["Margin Rank"] = (
            summary["Avg_Profit_Margin"].rank(ascending=False, method="min").astype(int)
        )
        rank_cols.append("Margin Rank")

    if not efficiency.empty and "Excellent" in efficiency.columns:
        efficiency = efficiency.copy()
        good_cols = [c for c in ["Excellent", "Good"] if c in efficiency.columns]
        if good_cols:
            efficiency["Eff_Score"] = efficiency[good_cols].sum(axis=1)
            ranking = ranking.merge(
                efficiency[[REGION_COL, "Eff_Score"]], on=REGION_COL, how="left"
            )
            ranking["Efficiency Rank"] = (
                ranking["Eff_Score"].rank(ascending=False, method="min").astype(int)
            )
            ranking = ranking.drop(columns=["Eff_Score"])
            rank_cols.append("Efficiency Rank")

    ranking["Overall Rank"] = (
        ranking[rank_cols]
        .mean(axis=1)
        .rank(ascending=True, method="min")
        .astype(int)
    )
    ranking = ranking.sort_values("Overall Rank").reset_index(drop=True)

    save_summary(ranking, "region_ranking.csv")
    print(f"  Regions ranked: {len(ranking)}")
    print(ranking.to_string(index=False))

    best_region  = ranking.iloc[0][REGION_COL]
    worst_region = ranking.iloc[-1][REGION_COL]
    add_insight(f"Best Overall Ranked Region (composite score): {best_region}")
    add_insight(f"Worst Overall Ranked Region (composite score): {worst_region}")

    return ranking


# ════════════════════════════════════════════════════════════════
# STEP 12 — CONSOLIDATED KPI SUMMARY
# ════════════════════════════════════════════════════════════════

def step12_kpi_summary(summary: pd.DataFrame,
                        efficiency: pd.DataFrame,
                        ranking: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single consolidated KPI table per region:
    Total Shipments, Sales, Profit, Avg Lead Time, Avg Profit Margin,
    Delay Rate %, Route Efficiency %, Overall Rank.
    """
    print("\n" + "=" * 60)
    print("STEP 12: Consolidated KPI Summary Table")
    print("=" * 60)

    kpi = summary[[
        REGION_COL, "Total_Shipments", "Total_Sales", "Total_Profit",
        "Avg_Lead_Time", "Avg_Profit_Margin",
    ]].copy()

    if "Delay_Rate_%" in summary.columns:
        kpi = kpi.merge(
            summary[[REGION_COL, "Delay_Rate_%"]], on=REGION_COL, how="left"
        )
    else:
        kpi["Delay_Rate_%"] = np.nan

    if not efficiency.empty:
        good_cols = [c for c in ["Excellent", "Good"] if c in efficiency.columns]
        if good_cols:
            eff_tmp = efficiency[[REGION_COL] + good_cols].copy()
            eff_tmp["Route_Efficiency_%"] = eff_tmp[good_cols].sum(axis=1).round(2)
            kpi = kpi.merge(
                eff_tmp[[REGION_COL, "Route_Efficiency_%"]], on=REGION_COL, how="left"
            )
        else:
            kpi["Route_Efficiency_%"] = np.nan
    else:
        kpi["Route_Efficiency_%"] = np.nan

    if not ranking.empty:
        kpi = kpi.merge(
            ranking[[REGION_COL, "Overall Rank"]], on=REGION_COL, how="left"
        )
    else:
        kpi["Overall Rank"] = np.nan

    kpi = kpi.rename(columns={
        REGION_COL:            "Region",
        "Total_Shipments":     "Total Shipments",
        "Total_Sales":         "Total Sales",
        "Total_Profit":        "Total Profit",
        "Avg_Lead_Time":       "Avg Lead Time",
        "Avg_Profit_Margin":   "Avg Profit Margin %",
        "Delay_Rate_%":        "Delay Rate %",
        "Route_Efficiency_%":  "Route Efficiency %",
    })
    kpi = kpi.sort_values("Overall Rank").reset_index(drop=True)

    save_summary(kpi, "region_kpi_summary.csv")
    print(kpi.to_string(index=False))

    return kpi


# ════════════════════════════════════════════════════════════════
# STEP 13 — CHART: TOTAL SALES BY REGION
# ════════════════════════════════════════════════════════════════

def step13_chart_sales(summary: pd.DataFrame) -> None:
    """Bar chart of Total Sales by Region."""
    print("\n" + "=" * 60)
    print("STEP 13: Chart — Total Sales by Region")
    print("=" * 60)

    ordered = summary.sort_values("Total_Sales", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ordered[REGION_COL], ordered["Total_Sales"],
                  color=sns.color_palette(PALETTE_CAT, len(ordered)), edgecolor="white")
    ax.set_title("Total Sales by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Sales ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"${h:,.0f}", (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9)

    save_chart("06_region_sales_bar.png")


# ════════════════════════════════════════════════════════════════
# STEP 14 — CHART: TOTAL GROSS PROFIT BY REGION
# ════════════════════════════════════════════════════════════════

def step14_chart_profit(summary: pd.DataFrame) -> None:
    """Bar chart of Total Gross Profit by Region."""
    print("\n" + "=" * 60)
    print("STEP 14: Chart — Total Gross Profit by Region")
    print("=" * 60)

    ordered = summary.sort_values("Total_Profit", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ordered[REGION_COL], ordered["Total_Profit"],
                  color=sns.color_palette("Greens_d", len(ordered)), edgecolor="white")
    ax.set_title("Total Gross Profit by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Gross Profit ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"${h:,.0f}", (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9)

    save_chart("06_region_profit_bar.png")


# ════════════════════════════════════════════════════════════════
# STEP 15 — CHART: SHIPMENT COUNT BY REGION
# ════════════════════════════════════════════════════════════════

def step15_chart_shipment_count(summary: pd.DataFrame) -> None:
    """Bar chart of shipment volume by Region."""
    print("\n" + "=" * 60)
    print("STEP 15: Chart — Shipment Count by Region")
    print("=" * 60)

    ordered = summary.sort_values("Total_Shipments", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ordered[REGION_COL], ordered["Total_Shipments"],
                  color=sns.color_palette(PALETTE_SEQ, len(ordered)), edgecolor="white")
    ax.set_title("Shipment Count by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Total Shipments")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h):,}", (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9)

    save_chart("06_region_shipment_count_bar.png")


# ════════════════════════════════════════════════════════════════
# STEP 16 — CHART: AVERAGE LEAD TIME BY REGION (BAR)
# ════════════════════════════════════════════════════════════════

def step16_chart_lead_time_bar(summary: pd.DataFrame) -> None:
    """Bar chart of Average Shipping Lead Time by Region."""
    print("\n" + "=" * 60)
    print("STEP 16: Chart — Average Lead Time by Region")
    print("=" * 60)

    ordered = summary.sort_values("Avg_Lead_Time", ascending=False)
    overall_avg = summary["Avg_Lead_Time"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ordered[REGION_COL], ordered["Avg_Lead_Time"],
                  color=sns.color_palette("Oranges_r", len(ordered)), edgecolor="white")
    ax.axhline(overall_avg, color="navy", linestyle="--", linewidth=1.5,
               label=f"Overall avg: {overall_avg:.1f}d")
    ax.set_title("Average Shipping Lead Time by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Avg Lead Time (days)")
    ax.legend()
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}d", (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9)

    save_chart("06_region_leadtime_bar.png")


# ════════════════════════════════════════════════════════════════
# STEP 17 — CHART: DELAY RATE BY REGION
# ════════════════════════════════════════════════════════════════

def step17_chart_delay_rate(summary: pd.DataFrame) -> None:
    """Bar chart of Delay Rate (%) by Region."""
    print("\n" + "=" * 60)
    print("STEP 17: Chart — Delay Rate by Region")
    print("=" * 60)

    if "Delay_Rate_%" not in summary.columns:
        print("  Delay_Rate_% column not found — skipping.")
        return

    ordered = summary.sort_values("Delay_Rate_%", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ordered[REGION_COL], ordered["Delay_Rate_%"],
                  color=sns.color_palette("Reds_r", len(ordered)), edgecolor="white")
    ax.set_title("Delay Rate (%) by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Delay Rate (%)")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9)

    save_chart("06_region_delay_rate_bar.png")


# ════════════════════════════════════════════════════════════════
# STEP 18 — CHART: AVERAGE PROFIT MARGIN BY REGION
# ════════════════════════════════════════════════════════════════

def step18_chart_profit_margin(summary: pd.DataFrame) -> None:
    """Bar chart of Average Profit Margin (%) by Region."""
    print("\n" + "=" * 60)
    print("STEP 18: Chart — Average Profit Margin by Region")
    print("=" * 60)

    ordered = summary.sort_values("Avg_Profit_Margin", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ordered[REGION_COL], ordered["Avg_Profit_Margin"],
                  color=sns.color_palette("YlGn", len(ordered)), edgecolor="white")
    ax.set_title("Average Profit Margin (%) by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Avg Profit Margin (%)")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9)

    save_chart("06_region_profit_margin_bar.png")


# ════════════════════════════════════════════════════════════════
# STEP 19 — CHART: ROUTE EFFICIENCY BY REGION (STACKED BAR)
# ════════════════════════════════════════════════════════════════

def step19_chart_route_efficiency(efficiency: pd.DataFrame) -> None:
    """Stacked bar chart of Route Efficiency Score distribution by Region."""
    print("\n" + "=" * 60)
    print("STEP 19: Chart — Route Efficiency by Region")
    print("=" * 60)

    if efficiency.empty:
        print("  Efficiency data not available — skipping.")
        return

    eff_cols = [c for c in EFF_SCORE_ORDER if c in efficiency.columns]
    eff_colors = {"Excellent": "#2ECC71", "Good": "#82E0AA",
                  "Average": "#F1C40F", "Poor": "#E74C3C"}
    colors = [eff_colors.get(c, "#95A5A6") for c in eff_cols]

    plot_df = efficiency.set_index(REGION_COL)[eff_cols]
    ax = plot_df.plot(kind="bar", stacked=True, color=colors,
                      figsize=(11, 6), edgecolor="white")
    ax.set_title("Route Efficiency Score Distribution by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Percentage (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Efficiency Score", bbox_to_anchor=(1.01, 1))

    save_chart("06_region_route_efficiency_stacked.png")


# ════════════════════════════════════════════════════════════════
# STEP 20 — CHART: STACKED BAR — REGION vs DELAY STATUS
# ════════════════════════════════════════════════════════════════

def step20_chart_delay_stacked(data: pd.DataFrame) -> None:
    """Stacked bar chart of raw Delay Status counts by Region."""
    print("\n" + "=" * 60)
    print("STEP 20: Chart — Region vs Delay Status (Stacked Bar)")
    print("=" * 60)

    if "Delay Status" not in data.columns:
        print("  Delay Status column not found — skipping.")
        return

    pivot = (
        data.groupby([REGION_COL, "Delay Status"])
        .size()
        .unstack(fill_value=0)
    )
    colors = [DELAY_STATUS_COLORS.get(c, "#95A5A6") for c in pivot.columns]

    ax = pivot.plot(kind="bar", stacked=True, color=colors,
                    figsize=(11, 6), edgecolor="white")
    ax.set_title("Delay Status Distribution by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Order Count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Delay Status")

    save_chart("06_region_delay_stacked.png")


# ════════════════════════════════════════════════════════════════
# STEP 21 — CHART: HEATMAP — REGION vs DELAY STATUS
# ════════════════════════════════════════════════════════════════

def step21_chart_heatmap_delay(data: pd.DataFrame) -> None:
    """Heatmap of Delay Status percentage share for each Region."""
    print("\n" + "=" * 60)
    print("STEP 21: Chart — Heatmap Region x Delay Status")
    print("=" * 60)

    if "Delay Status" not in data.columns:
        print("  Delay Status column not found — skipping.")
        return

    pivot = (
        data.groupby([REGION_COL, "Delay Status"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot_pct, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "% of Shipments"}, ax=ax,
    )
    ax.set_title("Region x Delay Status Heatmap (% of Shipments)", fontweight="bold")
    ax.set_xlabel("Delay Status")
    ax.set_ylabel("Region")
    save_chart("06_region_delay_heatmap.png")


# ════════════════════════════════════════════════════════════════
# STEP 22 — CHART: HEATMAP — REGION vs ROUTE EFFICIENCY
# ════════════════════════════════════════════════════════════════

def step22_chart_heatmap_efficiency(efficiency: pd.DataFrame) -> None:
    """Heatmap of Route Efficiency Score percentage share for each Region."""
    print("\n" + "=" * 60)
    print("STEP 22: Chart — Heatmap Region x Route Efficiency")
    print("=" * 60)

    if efficiency.empty:
        print("  Efficiency data not available — skipping.")
        return

    eff_cols = [c for c in EFF_SCORE_ORDER if c in efficiency.columns]
    subset = efficiency.set_index(REGION_COL)[eff_cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        subset, annot=True, fmt=".1f", cmap="RdYlGn",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "% of Shipments"}, ax=ax,
    )
    ax.set_title("Region x Route Efficiency Heatmap", fontweight="bold")
    ax.set_xlabel("Efficiency Score")
    ax.set_ylabel("Region")
    save_chart("06_region_efficiency_heatmap.png")


# ════════════════════════════════════════════════════════════════
# STEP 23 — CHART: BOXPLOT — LEAD TIME BY REGION
# ════════════════════════════════════════════════════════════════

def step23_chart_boxplot_lead_time(data: pd.DataFrame,
                                    summary: pd.DataFrame) -> None:
    """Box plot of Shipping Lead Time (Simulated) distribution by Region."""
    print("\n" + "=" * 60)
    print("STEP 23: Chart — Box Plot Lead Time by Region")
    print("=" * 60)

    order = summary.sort_values("Total_Shipments", ascending=False)[REGION_COL].tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=data, x=REGION_COL, y=LEAD_TIME_COL,
        order=order, palette=PALETTE_CAT, ax=ax,
    )
    ax.set_title("Lead Time Distribution by Region", fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Lead Time (days, Simulated)")
    save_chart("06_region_leadtime_boxplot.png")


# ════════════════════════════════════════════════════════════════
# STEP 24 — CHART: SCATTER PLOT — SALES vs PROFIT
# ════════════════════════════════════════════════════════════════

def step24_chart_scatter(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Scatter plot of order-level Sales vs Gross Profit, colored by Region."""
    print("\n" + "=" * 60)
    print("STEP 24: Chart — Scatter Plot Sales vs Profit")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(11, 8))
    regions = summary[REGION_COL].tolist()
    palette = dict(zip(regions, sns.color_palette(PALETTE_CAT, len(regions))))

    for region in regions:
        subset = data[data[REGION_COL] == region]
        ax.scatter(subset["Sales"], subset["Gross Profit"],
                   alpha=0.35, s=20, label=region, color=palette[region])

    ax.set_title("Sales vs Gross Profit by Region", fontweight="bold")
    ax.set_xlabel("Sales ($)")
    ax.set_ylabel("Gross Profit ($)")
    ax.legend(title="Region")
    save_chart("06_region_scatter_sales_profit.png")


# ════════════════════════════════════════════════════════════════
# STEP 25 — CHART: HISTOGRAM — SHIPPING LEAD TIME
# ════════════════════════════════════════════════════════════════

def step25_chart_histogram_leadtime(data: pd.DataFrame) -> None:
    """Histogram of the overall Shipping Lead Time (Simulated) distribution."""
    print("\n" + "=" * 60)
    print("STEP 25: Chart — Histogram Lead Time Distribution")
    print("=" * 60)

    series = pd.to_numeric(data[LEAD_TIME_COL], errors="coerce").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Shipping Lead Time (Simulated) — Distribution",
                 fontweight="bold", fontsize=14)

    axes[0].hist(series, bins=20, color=ACCENT_COLOR, edgecolor="white", alpha=0.85)
    axes[0].axvline(series.mean(), color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean: {series.mean():.1f}d")
    axes[0].axvline(series.median(), color="green", linestyle=":",
                    linewidth=1.5, label=f"Median: {series.median():.1f}d")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Lead Time (days)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].boxplot(series, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=ACCENT_COLOR, alpha=0.6),
                    medianprops=dict(color="red", linewidth=2))
    axes[1].set_title("Box Plot")
    axes[1].set_ylabel("Lead Time (days)")

    save_chart("06_region_leadtime_histogram.png")


# ════════════════════════════════════════════════════════════════
# STEP 26 — CHART: PIE — SHIPMENT SHARE BY REGION
# ════════════════════════════════════════════════════════════════

def step26_chart_pie_shipment_share(summary: pd.DataFrame) -> None:
    """Pie chart of shipment volume share by Region."""
    print("\n" + "=" * 60)
    print("STEP 26: Chart — Pie Shipment Share by Region")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        summary["Total_Shipments"],
        labels=summary[REGION_COL],
        autopct="%1.1f%%",
        colors=sns.color_palette(PALETTE_CAT, len(summary)),
        startangle=140,
    )
    ax.set_title("Shipment Volume Share by Region", fontweight="bold")
    save_chart("06_region_pie_shipment_share.png")


# ════════════════════════════════════════════════════════════════
# STEP 27 — CHART: LINE — AVERAGE LEAD TIME TREND
# ════════════════════════════════════════════════════════════════

def step27_chart_line_lead_time(summary: pd.DataFrame) -> None:
    """Line chart of Average Shipping Lead Time across Regions (ranked)."""
    print("\n" + "=" * 60)
    print("STEP 27: Chart — Line Average Lead Time")
    print("=" * 60)

    ordered = summary.sort_values("Avg_Lead_Time").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ordered[REGION_COL], ordered["Avg_Lead_Time"],
            marker="o", color=ACCENT_COLOR, linewidth=2.5, markersize=8)
    for x, y in zip(ordered[REGION_COL], ordered["Avg_Lead_Time"]):
        ax.annotate(f"{y:.1f}d", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title("Average Shipping Lead Time by Region (Fastest to Slowest)",
                 fontweight="bold")
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Lead Time (days)")
    ax.grid(alpha=0.3)
    save_chart("06_region_leadtime_line.png")


# ════════════════════════════════════════════════════════════════
# STEP 28 — CHARTS: TOP REGIONS BY SALES / PROFIT / ROUTE EFFICIENCY
# ════════════════════════════════════════════════════════════════

def step28_chart_top_regions(summary: pd.DataFrame,
                              kpi: pd.DataFrame) -> None:
    """
    Three ranked horizontal bar charts:
      - Top Regions by Sales
      - Top Regions by Profit
      - Top Regions by Route Efficiency %
    """
    print("\n" + "=" * 60)
    print("STEP 28: Charts — Top Regions by Sales / Profit / Efficiency")
    print("=" * 60)

    top_sales = summary.sort_values("Total_Sales", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_sales, y=REGION_COL, x="Total_Sales",
                palette="Blues_r", orient="h", ax=ax)
    ax.set_title("Top Regions by Sales", fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    save_chart("06_region_top_sales.png")

    top_profit = summary.sort_values("Total_Profit", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_profit, y=REGION_COL, x="Total_Profit",
                palette="Greens_r", orient="h", ax=ax)
    ax.set_title("Top Regions by Profit", fontweight="bold")
    ax.set_xlabel("Total Gross Profit ($)")
    save_chart("06_region_top_profit.png")

    if not kpi.empty and "Route Efficiency %" in kpi.columns:
        top_eff = kpi.sort_values("Route Efficiency %", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top_eff, y="Region", x="Route Efficiency %",
                    palette="RdYlGn", orient="h", ax=ax)
        ax.set_title("Top Regions by Route Efficiency %", fontweight="bold")
        ax.set_xlabel("Route Efficiency (%)")
        save_chart("06_region_top_route_efficiency.png")


# ════════════════════════════════════════════════════════════════
# STEP 29 — CHART: OVERALL KPI COMPARISON
# ════════════════════════════════════════════════════════════════

def step29_chart_kpi_overview(summary: pd.DataFrame,
                               ranking: pd.DataFrame) -> None:
    """
    Multi-metric bar chart comparing all regions across Sales, Profit,
    Lead Time and Delay Rate, ordered by Overall Rank.
    """
    print("\n" + "=" * 60)
    print("STEP 29: Chart — Overall KPI Comparison")
    print("=" * 60)

    if ranking.empty:
        print("  Ranking data not available — skipping.")
        return

    ordered_regions = ranking[REGION_COL].tolist()
    subset = summary.set_index(REGION_COL).reindex(ordered_regions).reset_index()

    metrics = {
        "Total_Sales":   ("Total Sales ($)",   ACCENT_COLOR),
        "Total_Profit":  ("Total Profit ($)",  GREEN_COLOR),
        "Avg_Lead_Time": ("Avg Lead Time (d)",  ORANGE_COLOR),
        "Delay_Rate_%":  ("Delay Rate (%)",     RED_COLOR),
    }
    metrics = {k: v for k, v in metrics.items() if k in subset.columns}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    fig.suptitle("KPI Comparison — All Regions (Ranked Best to Worst)",
                 fontweight="bold", fontsize=14)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, (label, color)) in zip(axes, metrics.items()):
        ax.barh(subset[REGION_COL], subset[col], color=color, edgecolor="white")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(label)
        ax.invert_yaxis()

    save_chart("06_region_kpi_overview.png")


# ════════════════════════════════════════════════════════════════
# STEP 30 — BUSINESS INSIGHTS ENGINE
# ════════════════════════════════════════════════════════════════

def step30_business_insights(summary: pd.DataFrame,
                              ranking: pd.DataFrame) -> None:
    """
    Auto-generate business insights covering best/worst performing
    region, highest sales/profit/shipment region, fastest/slowest
    shipping, highest/lowest delay, route efficiency and the greatest
    improvement opportunity.
    """
    print("\n" + "=" * 60)
    print("STEP 30: Business Insights")
    print("=" * 60)

    try:
        best_sales  = summary.loc[summary["Total_Sales"].idxmax()]
        worst_sales = summary.loc[summary["Total_Sales"].idxmin()]
        add_insight(f"Highest Sales Region: {best_sales[REGION_COL]} "
                    f"({fmt_currency(best_sales['Total_Sales'])})")
        add_insight(f"Lowest Sales Region: {worst_sales[REGION_COL]} "
                    f"({fmt_currency(worst_sales['Total_Sales'])})")

        best_profit  = summary.loc[summary["Total_Profit"].idxmax()]
        worst_profit = summary.loc[summary["Total_Profit"].idxmin()]
        add_insight(f"Highest Profit Region: {best_profit[REGION_COL]} "
                    f"({fmt_currency(best_profit['Total_Profit'])})")
        add_insight(f"Lowest Profit Region: {worst_profit[REGION_COL]} "
                    f"({fmt_currency(worst_profit['Total_Profit'])})")

        most_ship  = summary.loc[summary["Total_Shipments"].idxmax()]
        least_ship = summary.loc[summary["Total_Shipments"].idxmin()]
        total_ship = summary["Total_Shipments"].sum()
        add_insight(f"Highest Shipment Region: {most_ship[REGION_COL]} "
                    f"({int(most_ship['Total_Shipments']):,} orders = "
                    f"{most_ship['Total_Shipments'] / total_ship * 100:.1f}% of all)")
        add_insight(f"Lowest Shipment Region: {least_ship[REGION_COL]} "
                    f"({int(least_ship['Total_Shipments']):,} orders)")

        fastest = summary.loc[summary["Avg_Lead_Time"].idxmin()]
        slowest = summary.loc[summary["Avg_Lead_Time"].idxmax()]
        add_insight(f"Fastest Shipping Region: {fastest[REGION_COL]} "
                    f"({fastest['Avg_Lead_Time']:.2f} days avg lead time)")
        add_insight(f"Slowest Shipping Region: {slowest[REGION_COL]} "
                    f"({slowest['Avg_Lead_Time']:.2f} days avg lead time)")

        if "Delay_Rate_%" in summary.columns:
            highest_delay = summary.loc[summary["Delay_Rate_%"].idxmax()]
            lowest_delay  = summary.loc[summary["Delay_Rate_%"].idxmin()]
            add_insight(f"Highest Delay Region: {highest_delay[REGION_COL]} "
                        f"({fmt_pct(highest_delay['Delay_Rate_%'])} delayed)")
            add_insight(f"Lowest Delay Region: {lowest_delay[REGION_COL]} "
                        f"({fmt_pct(lowest_delay['Delay_Rate_%'])} delayed)")

        if "Avg_Profit_Margin" in summary.columns:
            best_margin = summary.loc[summary["Avg_Profit_Margin"].idxmax()]
            add_insight(f"Best Profit Margin Region: {best_margin[REGION_COL]} "
                        f"({fmt_pct(best_margin['Avg_Profit_Margin'])} avg margin)")

        if not ranking.empty and "Efficiency Rank" in ranking.columns:
            best_eff  = ranking.loc[ranking["Efficiency Rank"].idxmin()]
            worst_eff = ranking.loc[ranking["Efficiency Rank"].idxmax()]
            add_insight(f"Highest Route Efficiency Region: {best_eff[REGION_COL]}")
            add_insight(f"Lowest Route Efficiency Region: {worst_eff[REGION_COL]}")

        if not ranking.empty:
            add_insight(f"Best Overall Performing Region: {ranking.iloc[0][REGION_COL]}")

        # ── Improvement opportunity: high sales but high delay ────
        if "Delay_Rate_%" in summary.columns:
            opportunity = summary[summary["Delay_Rate_%"] > summary["Delay_Rate_%"].median()]
            opportunity = opportunity.sort_values("Total_Sales", ascending=False)
            if not opportunity.empty:
                top_opp = opportunity.iloc[0]
                add_insight(
                    f"Greatest Improvement Opportunity: {top_opp[REGION_COL]} - "
                    f"high sales ({fmt_currency(top_opp['Total_Sales'])}) but "
                    f"above-median delay rate ({fmt_pct(top_opp['Delay_Rate_%'])})"
                )

        add_insight(f"Combined Total Revenue (all regions): "
                    f"{fmt_currency(summary['Total_Sales'].sum())}")
        add_insight(f"Combined Total Gross Profit (all regions): "
                    f"{fmt_currency(summary['Total_Profit'].sum())}")
        add_insight(f"Overall Average Lead Time (all regions): "
                    f"{summary['Avg_Lead_Time'].mean():.2f} days")

    except Exception as exc:
        print(f"  WARNING: Insight generation encountered an issue: {exc}")

    insights_df = pd.DataFrame({
        "Insight #": range(1, len(insight_list) + 1),
        "Insight":   insight_list,
    })
    save_summary(insights_df, "region_business_insights.csv")
    print(f"\n  Total business insights generated: {len(insight_list)}")


# ════════════════════════════════════════════════════════════════
# STEP 31 — RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════

def step31_recommendation_engine(summary: pd.DataFrame,
                                  ranking: pd.DataFrame) -> str:
    """
    Auto-generate data-driven business recommendations covering
    Carrier Optimization, Route Optimization, Warehouse Planning,
    Inventory Distribution, Regional Expansion, Delay Reduction,
    Capacity Planning, Cost Reduction, Resource Allocation and
    Customer Service Improvement.
    """
    print("\n" + "=" * 60)
    print("STEP 31: Recommendation Engine")
    print("=" * 60)

    recs = []

    try:
        # 1. Carrier Optimization
        if "Avg_Lead_Time" in summary.columns:
            slow_regions = summary.nlargest(2, "Avg_Lead_Time")[REGION_COL].tolist()
            recs.append(
                "1. CARRIER OPTIMIZATION\n"
                f"   Renegotiate carrier SLAs for {', '.join(slow_regions)} — these "
                "regions record the longest average shipping lead times. Explore "
                "dedicated lane agreements or alternative carriers to close the gap "
                "with faster-performing regions."
            )

        # 2. Route Optimization
        if not ranking.empty and "Lead Time Rank" in ranking.columns:
            worst_lt = ranking.nlargest(2, "Lead Time Rank")[REGION_COL].tolist()
            recs.append(
                "2. ROUTE OPTIMIZATION\n"
                f"   Review factory-to-region routing for {', '.join(worst_lt)}. "
                "Shifting volume toward faster ship modes on underperforming routes, "
                "or introducing direct factory-to-hub lanes, can meaningfully cut "
                "lead time without disproportionate cost increases."
            )

        # 3. Warehouse Planning
        top_vol = summary.nlargest(2, "Total_Shipments")[REGION_COL].tolist()
        recs.append(
            "3. WAREHOUSE PLANNING\n"
            f"   Evaluate a regional distribution hub serving {', '.join(top_vol)} to "
            "reduce last-mile lead times. Positioning a hub within the highest-volume "
            "region cluster can simultaneously shorten delivery distances and lower "
            "cross-country freight costs."
        )

        # 4. Inventory Distribution
        top_sales_regions = summary.nlargest(2, "Total_Sales")[REGION_COL].tolist()
        recs.append(
            "4. INVENTORY DISTRIBUTION\n"
            f"   Prioritize safety stock allocation toward {', '.join(top_sales_regions)} "
            "— these regions generate the highest cumulative sales. Aligning inventory "
            "buffers with regional demand reduces stockout risk during peak periods."
        )

        # 5. Regional Expansion
        bottom_sales_regions = summary.nsmallest(2, "Total_Sales")[REGION_COL].tolist()
        recs.append(
            "5. REGIONAL EXPANSION\n"
            f"   Investigate low-penetration regions: {', '.join(bottom_sales_regions)}. "
            "Determine whether low volume reflects genuine low demand or insufficient "
            "distribution reach, and consider targeted promotions or lower shipping "
            "thresholds to stimulate growth."
        )

        # 6. Delay Reduction
        if "Delay_Rate_%" in summary.columns:
            high_delay = summary[summary["Delay_Rate_%"] > 30].nlargest(
                3, "Delay_Rate_%"
            )[REGION_COL].tolist()
            if high_delay:
                recs.append(
                    "6. DELAY REDUCTION\n"
                    f"   Regions with delay rates above 30%: {', '.join(high_delay)}. "
                    "Introduce real-time shipment tracking, build in buffer days for "
                    "Standard Class shipments into these regions, and set up proactive "
                    "exception alerts as orders approach SLA breach thresholds."
                )
            else:
                recs.append(
                    "6. DELAY REDUCTION\n"
                    "   No region currently exceeds a 30% delay rate. Maintain existing "
                    "SLA discipline with quarterly carrier performance reviews."
                )

        # 7. Capacity Planning
        recs.append(
            "7. CAPACITY PLANNING\n"
            "   Align fleet and carrier capacity with the highest-volume region to "
            "avoid bottlenecks during demand spikes, while preserving service levels "
            "on lower-volume regions through flexible surge-capacity agreements."
        )

        # 8. Cost Reduction
        if "Total_Cost" in summary.columns:
            high_cost = summary.nlargest(2, "Total_Cost")[REGION_COL].tolist()
            recs.append(
                "8. COST REDUCTION\n"
                f"   Cost concentration is highest in {', '.join(high_cost)}. Benchmark "
                "cost-per-unit against comparable routes and explore consolidation "
                "opportunities (e.g., full-truckload vs. LTL) to lower per-shipment "
                "costs in these regions."
            )

        # 9. Resource Allocation
        if not ranking.empty:
            top_ranked = ranking.head(2)[REGION_COL].tolist()
            recs.append(
                "9. RESOURCE ALLOCATION\n"
                f"   Direct field operations and account management resources toward "
                f"{', '.join(top_ranked)} — the top composite-ranked regions combining "
                "strong revenue, low delay rates, and high route efficiency. Protecting "
                "share in best-performing regions delivers the highest return on "
                "resource investment."
            )

        # 10. Customer Service Improvement
        if "Delay_Rate_%" in summary.columns:
            service_focus = summary.nlargest(2, "Delay_Rate_%")[REGION_COL].tolist()
            recs.append(
                "10. CUSTOMER SERVICE IMPROVEMENT\n"
                f"   Proactively communicate expected delivery windows to customers in "
                f"{', '.join(service_focus)}, where delay rates are highest. Automated "
                "shipment status notifications and simplified escalation paths can "
                "offset the customer experience impact while root-cause fixes are "
                "implemented."
            )

    except Exception as exc:
        print(f"  WARNING: Recommendation engine encountered an issue: {exc}")

    report_text = (
        "REGION-LEVEL BUSINESS RECOMMENDATIONS\n"
        "Factory-to-Customer Shipping Route Efficiency Analysis\n"
        "Nassau Candy Distributor\n"
        + "=" * 60 + "\n\n"
        + "\n\n".join(recs) + "\n"
    )

    save_report_text(report_text, "region_recommendations.txt")
    for rec in recs:
        print(f"\n  {rec}")

    return report_text


# ════════════════════════════════════════════════════════════════
# STEP 32 — EXECUTIVE REPORT
# ════════════════════════════════════════════════════════════════

def step32_executive_report(summary: pd.DataFrame,
                             ranking: pd.DataFrame,
                             anova_df: pd.DataFrame,
                             recommendations_text: str) -> None:
    """
    Generate a professional management-level executive report
    containing: Executive Summary, Key Findings, Regional Rankings,
    Statistical Results, Business Insights, Recommendations and a
    Conclusion. Saved as region_executive_report.txt.
    """
    print("\n" + "=" * 60)
    print("STEP 32: Executive Report")
    print("=" * 60)

    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        best_sales  = summary.loc[summary["Total_Sales"].idxmax()]
        best_profit = summary.loc[summary["Total_Profit"].idxmax()]
        fastest     = summary.loc[summary["Avg_Lead_Time"].idxmin()]
        slowest     = summary.loc[summary["Avg_Lead_Time"].idxmax()]

        lowest_delay_region  = "N/A"
        highest_delay_region = "N/A"
        if "Delay_Rate_%" in summary.columns:
            lowest_delay_region  = summary.loc[summary["Delay_Rate_%"].idxmin(), REGION_COL]
            highest_delay_region = summary.loc[summary["Delay_Rate_%"].idxmax(), REGION_COL]

        best_overall = ranking.iloc[0][REGION_COL] if not ranking.empty else "N/A"

        lines = [
            "=" * 60,
            "EXECUTIVE REPORT - REGION-LEVEL SHIPPING PERFORMANCE",
            "Factory-to-Customer Shipping Route Efficiency Analysis",
            "Nassau Candy Distributor",
            f"Generated: {ts}",
            "=" * 60,
            "",
            section("EXECUTIVE SUMMARY").strip(),
            (
                f"This report presents a comprehensive region-level analysis of "
                f"shipping performance across {len(summary)} regions served by "
                f"Nassau Candy Distributor. The analysis covers total sales, gross "
                f"profit, shipment volume, shipping lead time, delay rates, route "
                f"efficiency and profit margin, and concludes with data-driven "
                f"business recommendations. The region ranked #1 overall on a "
                f"composite performance score is {best_overall}."
            ),
            "",
            section("KEY FINDINGS").strip(),
            f"  - Highest Sales Region        : {best_sales[REGION_COL]} "
            f"({fmt_currency(best_sales['Total_Sales'])})",
            f"  - Highest Profit Region       : {best_profit[REGION_COL]} "
            f"({fmt_currency(best_profit['Total_Profit'])})",
            f"  - Fastest Shipping Region     : {fastest[REGION_COL]} "
            f"({fastest['Avg_Lead_Time']:.2f} days avg)",
            f"  - Slowest Shipping Region     : {slowest[REGION_COL]} "
            f"({slowest['Avg_Lead_Time']:.2f} days avg)",
            f"  - Lowest Delay Region         : {lowest_delay_region}",
            f"  - Highest Delay Region        : {highest_delay_region}",
            f"  - Best Overall Region         : {best_overall}",
            f"  - Total Revenue (all regions) : "
            f"{fmt_currency(summary['Total_Sales'].sum())}",
            f"  - Total Gross Profit          : "
            f"{fmt_currency(summary['Total_Profit'].sum())}",
            f"  - Overall Avg Lead Time       : "
            f"{summary['Avg_Lead_Time'].mean():.2f} days",
        ]

        if not ranking.empty:
            lines += ["", section("REGIONAL RANKINGS").strip()]
            for _, row in ranking.iterrows():
                lines.append(f"  Rank {int(row['Overall Rank'])}: {row[REGION_COL]}")

        if not anova_df.empty:
            lines += ["", section("STATISTICAL RESULTS (ANOVA)").strip()]
            for _, row in anova_df.iterrows():
                lines.append(
                    f"  - {row['Metric']}: F={row['F-Statistic']}, "
                    f"p={row['P-Value']} "
                    f"(Region effect significant: {row['Significant (p<0.05)']})"
                )

        lines += ["", section("BUSINESS INSIGHTS").strip()]
        for insight in insight_list:
            lines.append(f"  - {insight}")

        lines += [
            "",
            section("RECOMMENDATIONS").strip(),
            recommendations_text.split("=" * 60, 1)[-1].strip(),
            "",
            section("CONCLUSION").strip(),
            textwrap.fill(
                "Regional shipping performance varies meaningfully across Nassau "
                "Candy Distributor's footprint. By concentrating resources on "
                "high-performing regions, addressing delay hotspots with targeted "
                "carrier and route interventions, and aligning warehouse and "
                "inventory strategy with regional demand patterns, the company can "
                "materially improve delivery reliability, control logistics costs, "
                "and support sustainable revenue growth across every region it serves.",
                width=68,
            ),
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ]

        report_text = "\n".join(lines) + "\n"
        save_report_text(report_text, "region_executive_report.txt")

    except Exception as exc:
        print(f"  WARNING: Executive report generation failed: {exc}")


# ════════════════════════════════════════════════════════════════
# STEP 33 — DASHBOARD-READY DATASETS
# ════════════════════════════════════════════════════════════════

def step33_dashboard_datasets(data: pd.DataFrame,
                               kpi: pd.DataFrame) -> None:
    """
    Build two lightweight datasets for the Streamlit dashboard's
    Region Analysis module:
      dashboard_region.csv     - row-level data with key columns only
      dashboard_region_kpi.csv - aggregated KPI table per region
    """
    print("\n" + "=" * 60)
    print("STEP 33: Dashboard-Ready Datasets")
    print("=" * 60)

    dashboard_cols = [
        "Order ID", REGION_COL, "State/Province", "City", "Order Date",
        "Factory", "Ship Mode", "Sales", "Gross Profit",
        "Profit Margin %", LEAD_TIME_COL, "Delay Status",
        "Route Efficiency Score", "Units", "Cost",
    ]
    dashboard_cols = [c for c in dashboard_cols if c in data.columns]

    try:
        dash_df = data[dashboard_cols].copy()
        dash_df = dash_df.dropna(subset=[REGION_COL])
        save_summary(dash_df, "dashboard_region.csv")

        if not kpi.empty:
            save_summary(kpi, "dashboard_region_kpi.csv")

        print(f"  Row-level dataset shape : {dash_df.shape}")
        print(f"  KPI dataset shape       : {kpi.shape if not kpi.empty else 'N/A'}")
        print(f"  Columns in row dataset  : {dashboard_cols}")

    except Exception as exc:
        print(f"  WARNING: Dashboard dataset generation failed: {exc}")


# ════════════════════════════════════════════════════════════════
# STEP 34 — FINAL EXECUTION REPORT
# ════════════════════════════════════════════════════════════════

def step34_final_report() -> None:
    """
    Print a complete execution summary showing total charts, summary
    tables, text reports and business insights generated.
    """
    print("\n" + "=" * 60)
    print("STEP 34: Final Execution Report")
    print("=" * 60)
    print(f"  Total Charts Generated      : {chart_count}")
    print(f"  Total Summary Tables Saved  : {summary_count}")
    print(f"  Total Text Reports Saved    : {report_count}")
    print(f"  Total Business Insights     : {len(insight_list)}")
    print(f"  Charts saved in             : ./{CHARTS_DIR}/")
    print(f"  Summaries & reports saved in: ./{SUMMARIES_DIR}/")
    print("  Region-Level Analysis Completed Successfully")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """Execute the complete region-level analysis pipeline."""
    try:
        # ── Data loading & quality ────────────────────────────
        df = step1_load_dataset()
        step2_data_quality(df)

        # ── Core aggregations ─────────────────────────────────
        summary    = step3_build_region_summary(df)
        efficiency = step4_route_efficiency_by_region(df)
        step5_region_comparison(summary, efficiency)
        step6_delay_summary(df)

        # ── Statistical analysis ──────────────────────────────
        step7_statistical_summary(df)
        step8_outlier_detection(df)
        step9_correlation_analysis(df)
        anova_df = step10_anova_test(df)

        # ── Ranking & KPI ─────────────────────────────────────
        ranking = step11_region_ranking(summary, efficiency)
        kpi     = step12_kpi_summary(summary, efficiency, ranking)

        # ── Visualizations (Charts 1-20 as specified) ─────────
        step13_chart_sales(summary)
        step14_chart_profit(summary)
        step15_chart_shipment_count(summary)
        step16_chart_lead_time_bar(summary)
        step17_chart_delay_rate(summary)
        step18_chart_profit_margin(summary)
        step19_chart_route_efficiency(efficiency)
        step20_chart_delay_stacked(df)
        step21_chart_heatmap_delay(df)
        step22_chart_heatmap_efficiency(efficiency)
        step23_chart_boxplot_lead_time(df, summary)
        step24_chart_scatter(df, summary)
        # (Correlation Heatmap is generated inside step9_correlation_analysis, run earlier)
        step25_chart_histogram_leadtime(df)
        step26_chart_pie_shipment_share(summary)
        step27_chart_line_lead_time(summary)
        step28_chart_top_regions(summary, kpi)
        step29_chart_kpi_overview(summary, ranking)

        # ── Insights, recommendations & reports ───────────────
        step30_business_insights(summary, ranking)
        recs_text = step31_recommendation_engine(summary, ranking)
        step32_executive_report(summary, ranking, anova_df, recs_text)

        # ── Dashboard datasets & final report ─────────────────
        step33_dashboard_datasets(df, kpi)
        step34_final_report()

    except Exception as exc:
        print(f"\nPIPELINE FAILED: {exc}")
        raise


if __name__ == "__main__":
    main()