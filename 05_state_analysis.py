"""
05_state_analysis.py
--------------------------
Phase 5 - State-Level Shipping Performance Analysis
Project: Factory-to-Customer Shipping Route Efficiency Analysis
         for Nassau Candy Distributor

Input  : featured_nassau_candy.csv  (output of 02_feature_engineering.py)
Output : EDA_Charts/      (state-level visualizations)
         EDA_Summaries/   (state summary tables + reports)

Run after:
  01_data_cleaning.py
  02_feature_engineering.py
  03_route_analysis.py
  04_ship_mode_analysis.py
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
    "figure.figsize": (14, 7),
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

TOP_N  = 10   # default top/bottom N states for charts
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
STATE_COL     = "State/Province"

# ─────────────────────────────────────────────────────────────
# COUNTERS & GLOBAL ACCUMULATORS
# ─────────────────────────────────────────────────────────────
chart_count  = 0
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


def top_n_states(data: pd.DataFrame, col: str, n: int = TOP_N,
                 ascending: bool = False) -> pd.DataFrame:
    """Return the top-N (or bottom-N) state rows ranked by *col*."""
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
        STATE_COL, "Sales", "Gross Profit", "Profit Margin %",
        LEAD_TIME_COL, "Delay Status", "Route Efficiency Score",
        "Order ID", "Region", "Factory",
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
    print(f"Unique States       : {data[STATE_COL].nunique()}")
    print(f"Unique Regions      : {data['Region'].nunique()}")
    print(f"Date Range          : {data['Order Date'].min()} — {data['Order Date'].max()}")

    return data


# ════════════════════════════════════════════════════════════════
# STEP 2 — DATA QUALITY CHECK
# ════════════════════════════════════════════════════════════════

def step2_data_quality(data: pd.DataFrame) -> None:
    """
    Check for missing values, duplicate rows and confirm that the
    key state-analysis columns are complete.
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

    us_count = data.get("Is US Record", pd.Series([True] * len(data))).sum()
    ca_count = len(data) - us_count
    print(f"\n  US Records      : {int(us_count):,}")
    print(f"  Canadian Records: {int(ca_count):,}")
    print(
        "  NOTE: Both US and Canadian records are included in state/province "
        "analysis. US-only charts are labelled accordingly."
    )


# ════════════════════════════════════════════════════════════════
# STEP 3 — BUILD CORE STATE SUMMARY
# ════════════════════════════════════════════════════════════════

def step3_build_state_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate key KPIs per state:
      Total Shipments, Total Sales, Total Gross Profit, Total Units,
      Total Cost, Average Lead Time, Average Profit Margin,
      On-Time Rate %, Moderate Delay %, Delay Rate %.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Core State Summary")
    print("=" * 60)

    summary = (
        data.groupby(STATE_COL)
        .agg(
            Total_Shipments=("Order ID",    "count"),
            Total_Sales=    ("Sales",       "sum"),
            Total_Profit=   ("Gross Profit","sum"),
            Total_Units=    ("Units",       "sum"),
            Total_Cost=     ("Cost",        "sum"),
            Avg_Lead_Time=  (LEAD_TIME_COL, "mean"),
            Avg_Profit_Margin=("Profit Margin %", "mean"),
        )
        .round(2)
        .reset_index()
    )

    # ── Delay Status breakdown per state ──────────────────────
    if "Delay Status" in data.columns:
        status_counts = (
            data.groupby([STATE_COL, "Delay Status"])
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
            status_pct[[STATE_COL, "On_Time_Rate_%",
                         "Moderate_Delay_%", "Delay_Rate_%"]],
            on=STATE_COL, how="left",
        )
    else:
        summary["On_Time_Rate_%"]    = np.nan
        summary["Moderate_Delay_%"]  = np.nan
        summary["Delay_Rate_%"]      = np.nan

    # ── Dominant Region per state ──────────────────────────────
    if "Region" in data.columns:
        dom_region = (
            data.groupby(STATE_COL)["Region"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
            .rename("Region")
            .reset_index()
        )
        summary = summary.merge(dom_region, on=STATE_COL, how="left")

    summary = summary.fillna(0)
    summary = summary.sort_values("Total_Sales", ascending=False).reset_index(drop=True)

    save_summary(summary, "state_summary.csv")
    print(f"  States captured : {len(summary)}")
    print(summary.head(10).to_string(index=False))

    return summary


# ════════════════════════════════════════════════════════════════
# STEP 4 — ROUTE EFFICIENCY BY STATE
# ════════════════════════════════════════════════════════════════

def step4_route_efficiency_by_state(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Route Efficiency Score percentage distribution
    (Excellent / Good / Average / Poor) for each state.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Route Efficiency by State")
    print("=" * 60)

    if "Route Efficiency Score" not in data.columns:
        print("  Route Efficiency Score column not found — skipping.")
        return pd.DataFrame()

    counts = (
        data.groupby([STATE_COL, "Route Efficiency Score"])
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

    save_summary(pct, "state_route_efficiency.csv")
    print(pct.head(10).to_string(index=False))

    return pct


# ════════════════════════════════════════════════════════════════
# STEP 5 — STATE COMPARISON TABLE
# ════════════════════════════════════════════════════════════════

def step5_state_comparison(summary: pd.DataFrame,
                            efficiency: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the core state summary with Route Efficiency Score
    distribution into one consolidated comparison table.
    """
    print("\n" + "=" * 60)
    print("STEP 5: State Comparison Table")
    print("=" * 60)

    comparison = summary.copy()
    if not efficiency.empty:
        comparison = comparison.merge(efficiency, on=STATE_COL, how="left")

    save_summary(comparison, "state_comparison.csv")
    print(f"  Comparison table shape: {comparison.shape}")

    return comparison


# ════════════════════════════════════════════════════════════════
# STEP 6 — DELAY SUMMARY BY STATE
# ════════════════════════════════════════════════════════════════

def step6_delay_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build a detailed delay summary per state: raw counts and
    percentage for each Delay Status category.
    """
    print("\n" + "=" * 60)
    print("STEP 6: Delay Summary by State")
    print("=" * 60)

    if "Delay Status" not in data.columns:
        print("  Delay Status column not found — skipping.")
        return pd.DataFrame()

    counts = (
        data.groupby([STATE_COL, "Delay Status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    totals = counts.set_index(STATE_COL).sum(axis=1).rename("Total")
    counts = counts.set_index(STATE_COL)
    for col in ["On Time", "Moderate Delay", "Delayed"]:
        if col in counts.columns:
            counts[f"{col} %"] = (counts[col] / totals * 100).round(2)
    counts = counts.reset_index()

    counts = counts.sort_values("Delayed", ascending=False) \
                   if "Delayed" in counts.columns \
                   else counts
    counts = counts.reset_index(drop=True)

    save_summary(counts, "state_delay_summary.csv")
    print(counts.head(10).to_string(index=False))

    return counts


# ════════════════════════════════════════════════════════════════
# STEP 7 — STATISTICAL SUMMARY
# ════════════════════════════════════════════════════════════════

def step7_statistical_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Mean, Median, Std Dev, Variance, Min, Max, Q1, Q3, IQR
    for Sales, Gross Profit and Shipping Lead Time, broken down by
    state.
    """
    print("\n" + "=" * 60)
    print("STEP 7: Statistical Summary by State")
    print("=" * 60)

    target_cols = ["Sales", "Gross Profit", LEAD_TIME_COL]
    target_cols = [c for c in target_cols if c in data.columns]

    stat_rows = []
    for state, grp in data.groupby(STATE_COL):
        for col in target_cols:
            series = pd.to_numeric(grp[col], errors="coerce").dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr    = q3 - q1
            stat_rows.append({
                STATE_COL:  state,
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
    save_summary(stats_df, "state_statistics.csv")
    print(f"  Statistical records generated: {len(stats_df):,}")
    print(stats_df[stats_df["Metric"] == "Sales"].head(6).to_string(index=False))

    return stats_df


# ════════════════════════════════════════════════════════════════
# STEP 8 — OUTLIER DETECTION (IQR METHOD)
# ════════════════════════════════════════════════════════════════

def step8_outlier_detection(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers in Sales, Gross Profit and Shipping Lead Time
    using the 1.5 × IQR method, reported per state.
    """
    print("\n" + "=" * 60)
    print("STEP 8: Outlier Detection (IQR Method)")
    print("=" * 60)

    target_cols = ["Sales", "Gross Profit", LEAD_TIME_COL]
    target_cols = [c for c in target_cols if c in data.columns]

    outlier_rows = []
    for col in target_cols:
        series = pd.to_numeric(data[col], errors="coerce")
        q1, q3   = series.quantile([0.25, 0.75])
        iqr      = q3 - q1
        lower    = q1 - 1.5 * iqr
        upper    = q3 + 1.5 * iqr
        is_out   = (series < lower) | (series > upper)

        temp = data.loc[series.notna()].copy()
        temp["__out__"] = is_out.loc[series.notna()]

        for state, grp in temp.groupby(STATE_COL):
            total    = len(grp)
            n_out    = int(grp["__out__"].sum())
            outlier_rows.append({
                STATE_COL:       state,
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

    save_summary(outliers_df, "state_outliers.csv")
    print(f"  Outlier records detected: {outliers_df['Outlier Count'].sum():,}")
    print(outliers_df.head(8).to_string(index=False))

    return outliers_df


# ════════════════════════════════════════════════════════════════
# STEP 9 — CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════

def step9_correlation_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and visualize a correlation matrix for the key numeric
    columns: Sales, Gross Profit, Profit Margin %, Lead Time, Units.
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
        "state_correlation.csv",
    )
    print(corr_matrix.to_string())

    # ── Correlation Heatmap ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Pearson r"}, ax=ax,
    )
    ax.set_title("Correlation Heatmap — Key State Metrics", fontweight="bold")
    save_chart("05_state_correlation_heatmap.png")

    return corr_matrix


# ════════════════════════════════════════════════════════════════
# STEP 10 — ANOVA TEST
# ════════════════════════════════════════════════════════════════

def step10_anova_test(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a one-way ANOVA to determine whether the state a shipment
    originates from / is delivered to significantly affects Sales,
    Gross Profit and Shipping Lead Time.
    """
    print("\n" + "=" * 60)
    print("STEP 10: ANOVA Test (State Effect)")
    print("=" * 60)

    target_cols = ["Sales", "Gross Profit", LEAD_TIME_COL]
    target_cols = [c for c in target_cols if c in data.columns]

    anova_rows = []
    for col in target_cols:
        try:
            groups = [
                pd.to_numeric(grp[col], errors="coerce").dropna()
                for _, grp in data.groupby(STATE_COL)
            ]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                continue
            f_stat, p_val = stats.f_oneway(*groups)
            anova_rows.append({
                "Metric":               col,
                "F-Statistic":          round(f_stat, 4),
                "P-Value":              round(p_val,  6),
                "Significant (p<0.05)": "Yes" if p_val < 0.05 else "No",
            })
        except Exception as exc:
            print(f"  WARNING: ANOVA failed for '{col}': {exc}")

    anova_df = pd.DataFrame(anova_rows)

    if not anova_df.empty:
        save_summary(anova_df, "state_anova_results.csv")
        print(anova_df.to_string(index=False))
        for _, row in anova_df.iterrows():
            add_insight(
                f"ANOVA — {row['Metric']}: F={row['F-Statistic']}, "
                f"p={row['P-Value']} → State effect significant: "
                f"{row['Significant (p<0.05)']}"
            )
    else:
        print("  WARNING: No ANOVA results generated.")

    return anova_df


# ════════════════════════════════════════════════════════════════
# STEP 11 — STATE RANKING
# ════════════════════════════════════════════════════════════════

def step11_state_ranking(summary: pd.DataFrame,
                          efficiency: pd.DataFrame) -> pd.DataFrame:
    """
    Rank every state on Sales, Profit, Lead Time, Delay Rate and
    Route Efficiency, then compute an Overall Rank from the average
    of the individual rank scores (lower = better).
    """
    print("\n" + "=" * 60)
    print("STEP 11: State Ranking")
    print("=" * 60)

    ranking = summary[[STATE_COL]].copy()

    ranking["Sales Rank"]  = (
        summary["Total_Sales"].rank(ascending=False, method="min").astype(int)
    )
    ranking["Profit Rank"] = (
        summary["Total_Profit"].rank(ascending=False, method="min").astype(int)
    )
    ranking["Lead Time Rank"] = (
        summary["Avg_Lead_Time"].rank(ascending=True, method="min").astype(int)
    )

    rank_cols = ["Sales Rank", "Profit Rank", "Lead Time Rank"]

    if "Delay_Rate_%" in summary.columns:
        ranking["Delay Rank"] = (
            summary["Delay_Rate_%"].rank(ascending=True, method="min").astype(int)
        )
        rank_cols.append("Delay Rank")

    if not efficiency.empty and "Excellent" in efficiency.columns:
        efficiency = efficiency.copy()
        good_cols  = [c for c in ["Excellent", "Good"] if c in efficiency.columns]
        if good_cols:
            efficiency["Eff_Score"] = efficiency[good_cols].sum(axis=1)
            ranking = ranking.merge(
                efficiency[[STATE_COL, "Eff_Score"]], on=STATE_COL, how="left"
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

    save_summary(ranking, "state_ranking.csv")
    print(f"  States ranked: {len(ranking)}")
    print(ranking.head(10).to_string(index=False))

    top_state  = ranking.iloc[0][STATE_COL]
    worst_state = ranking.iloc[-1][STATE_COL]
    add_insight(f"Best Overall Ranked State (composite score): {top_state}")
    add_insight(f"Worst Overall Ranked State (composite score): {worst_state}")

    return ranking


# ════════════════════════════════════════════════════════════════
# STEP 12 — CONSOLIDATED KPI SUMMARY
# ════════════════════════════════════════════════════════════════

def step12_kpi_summary(summary: pd.DataFrame,
                        efficiency: pd.DataFrame,
                        ranking: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single consolidated KPI table per state:
    Total Shipments, Sales, Profit, Avg Lead Time, Avg Profit Margin,
    Delay Rate %, Route Efficiency %, Overall Rank.
    """
    print("\n" + "=" * 60)
    print("STEP 12: Consolidated KPI Summary Table")
    print("=" * 60)

    kpi = summary[[
        STATE_COL, "Total_Shipments", "Total_Sales", "Total_Profit",
        "Avg_Lead_Time", "Avg_Profit_Margin",
    ]].copy()

    if "Delay_Rate_%" in summary.columns:
        kpi = kpi.merge(
            summary[[STATE_COL, "Delay_Rate_%"]], on=STATE_COL, how="left"
        )
    else:
        kpi["Delay_Rate_%"] = np.nan

    if not efficiency.empty:
        good_cols = [c for c in ["Excellent", "Good"] if c in efficiency.columns]
        if good_cols:
            eff_tmp = efficiency[[STATE_COL] + good_cols].copy()
            eff_tmp["Route_Efficiency_%"] = eff_tmp[good_cols].sum(axis=1).round(2)
            kpi = kpi.merge(
                eff_tmp[[STATE_COL, "Route_Efficiency_%"]], on=STATE_COL, how="left"
            )
        else:
            kpi["Route_Efficiency_%"] = np.nan
    else:
        kpi["Route_Efficiency_%"] = np.nan

    if not ranking.empty:
        kpi = kpi.merge(
            ranking[[STATE_COL, "Overall Rank"]], on=STATE_COL, how="left"
        )
    else:
        kpi["Overall Rank"] = np.nan

    kpi = kpi.rename(columns={
        STATE_COL:             "State",
        "Total_Shipments":     "Total Shipments",
        "Total_Sales":         "Total Sales",
        "Total_Profit":        "Total Profit",
        "Avg_Lead_Time":       "Avg Lead Time",
        "Avg_Profit_Margin":   "Avg Profit Margin %",
        "Delay_Rate_%":        "Delay Rate %",
        "Route_Efficiency_%":  "Route Efficiency %",
    })
    kpi = kpi.sort_values("Overall Rank").reset_index(drop=True)

    save_summary(kpi, "state_kpi_summary.csv")
    print(kpi.head(10).to_string(index=False))

    return kpi


# ════════════════════════════════════════════════════════════════
# STEP 13 — CHART: TOTAL SALES BY STATE (TOP 20 HORIZONTAL BAR)
# ════════════════════════════════════════════════════════════════

def step13_chart_sales_by_state(summary: pd.DataFrame) -> None:
    """Horizontal bar chart — Total Sales for the Top 20 states."""
    print("\n" + "=" * 60)
    print("STEP 13: Chart — Total Sales by State")
    print("=" * 60)

    top20 = top_n_states(summary, "Total_Sales", n=20)

    fig, ax = plt.subplots(figsize=(13, 10))
    bars = ax.barh(top20[STATE_COL], top20["Total_Sales"],
                   color=ACCENT_COLOR, edgecolor="white")
    ax.set_title("Top 20 States by Total Sales", fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    ax.set_ylabel("State / Province")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"${w:,.0f}", va="center", ha="left", fontsize=8)
    save_chart("05_state_sales_bar.png")


# ════════════════════════════════════════════════════════════════
# STEP 14 — CHART: TOTAL PROFIT BY STATE (TOP 20 HORIZONTAL BAR)
# ════════════════════════════════════════════════════════════════

def step14_chart_profit_by_state(summary: pd.DataFrame) -> None:
    """Horizontal bar chart — Total Gross Profit for the Top 20 states."""
    print("\n" + "=" * 60)
    print("STEP 14: Chart — Total Profit by State")
    print("=" * 60)

    top20 = top_n_states(summary, "Total_Profit", n=20)

    fig, ax = plt.subplots(figsize=(13, 10))
    bars = ax.barh(top20[STATE_COL], top20["Total_Profit"],
                   color=GREEN_COLOR, edgecolor="white")
    ax.set_title("Top 20 States by Total Gross Profit", fontweight="bold")
    ax.set_xlabel("Total Gross Profit ($)")
    ax.set_ylabel("State / Province")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"${w:,.0f}", va="center", ha="left", fontsize=8)
    save_chart("05_state_profit_bar.png")


# ════════════════════════════════════════════════════════════════
# STEP 15 — CHART: AVERAGE LEAD TIME BY STATE
# ════════════════════════════════════════════════════════════════

def step15_chart_lead_time(summary: pd.DataFrame) -> None:
    """
    Two horizontal bar charts side by side:
      (a) Top 15 Fastest States  (b) Top 15 Slowest States.
    """
    print("\n" + "=" * 60)
    print("STEP 15: Chart — Average Lead Time by State")
    print("=" * 60)

    fastest = top_n_states(summary, "Avg_Lead_Time", n=15, ascending=True)
    slowest = top_n_states(summary, "Avg_Lead_Time", n=15, ascending=False)
    overall_avg = summary["Avg_Lead_Time"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Average Shipping Lead Time by State", fontweight="bold", fontsize=15)

    for ax, df_plot, title, palette in [
        (axes[0], fastest, "Top 15 Fastest States", "Greens_r"),
        (axes[1], slowest, "Top 15 Slowest States", "Reds_r"),
    ]:
        sns.barplot(data=df_plot, y=STATE_COL, x="Avg_Lead_Time",
                    palette=palette, orient="h", ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Avg Lead Time (days)")
        ax.set_ylabel("State / Province")
        ax.axvline(overall_avg, color="navy", linestyle="--", linewidth=1.5,
                   label=f"Overall avg: {overall_avg:.1f}d")
        ax.legend(fontsize=8)

    save_chart("05_state_lead_time.png")


# ════════════════════════════════════════════════════════════════
# STEP 16 — CHART: DELAY RATE BY STATE (TOP & BOTTOM 15)
# ════════════════════════════════════════════════════════════════

def step16_chart_delay_rate(summary: pd.DataFrame) -> None:
    """Horizontal bar charts for the 15 highest- and lowest-delay states."""
    print("\n" + "=" * 60)
    print("STEP 16: Chart — Delay Rate by State")
    print("=" * 60)

    if "Delay_Rate_%" not in summary.columns:
        print("  Delay_Rate_% column not found — skipping.")
        return

    highest = top_n_states(summary, "Delay_Rate_%", n=15)
    lowest  = top_n_states(summary, "Delay_Rate_%", n=15, ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Delay Rate (%) by State", fontweight="bold", fontsize=15)

    sns.barplot(data=highest, y=STATE_COL, x="Delay_Rate_%",
                palette="Reds_r", orient="h", ax=axes[0])
    axes[0].set_title("15 Highest Delay Rate States", fontsize=12)
    axes[0].set_xlabel("Delay Rate (%)")

    sns.barplot(data=lowest, y=STATE_COL, x="Delay_Rate_%",
                palette="Greens_r", orient="h", ax=axes[1])
    axes[1].set_title("15 Lowest Delay Rate States", fontsize=12)
    axes[1].set_xlabel("Delay Rate (%)")

    save_chart("05_state_delay_rate.png")


# ════════════════════════════════════════════════════════════════
# STEP 17 — CHART: SHIPMENT COUNT BY STATE
# ════════════════════════════════════════════════════════════════

def step17_chart_shipment_count(summary: pd.DataFrame) -> None:
    """Horizontal bar chart — Top 20 states by number of shipments."""
    print("\n" + "=" * 60)
    print("STEP 17: Chart — Shipment Count by State")
    print("=" * 60)

    top20 = top_n_states(summary, "Total_Shipments", n=20)

    fig, ax = plt.subplots(figsize=(13, 10))
    colors = sns.color_palette(PALETTE_SEQ, len(top20))
    bars   = ax.barh(top20[STATE_COL], top20["Total_Shipments"],
                     color=colors, edgecolor="white")
    ax.set_title("Top 20 States by Shipment Count", fontweight="bold")
    ax.set_xlabel("Total Shipments")
    ax.set_ylabel("State / Province")
    ax.invert_yaxis()
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", ha="left", fontsize=8)
    save_chart("05_state_shipment_count.png")


# ════════════════════════════════════════════════════════════════
# STEP 18 — CHART: ROUTE EFFICIENCY BY STATE (STACKED BAR)
# ════════════════════════════════════════════════════════════════

def step18_chart_route_efficiency(efficiency: pd.DataFrame) -> None:
    """
    Stacked horizontal bar chart of Route Efficiency Score distribution
    for the top 20 states by total shipments.
    """
    print("\n" + "=" * 60)
    print("STEP 18: Chart — Route Efficiency by State")
    print("=" * 60)

    if efficiency.empty:
        print("  Efficiency data not available — skipping.")
        return

    top20_states = efficiency.head(20).copy()
    eff_cols = [c for c in EFF_SCORE_ORDER if c in top20_states.columns]

    eff_colors = {
        "Excellent": "#2ECC71",
        "Good":      "#82E0AA",
        "Average":   "#F1C40F",
        "Poor":      "#E74C3C",
    }
    colors = [eff_colors.get(c, "#95A5A6") for c in eff_cols]

    plot_df = top20_states.set_index(STATE_COL)[eff_cols]
    ax = plot_df.plot(kind="barh", stacked=True, color=colors,
                      figsize=(13, 10), edgecolor="white")
    ax.set_title("Route Efficiency Score Distribution by State (Top 20)",
                 fontweight="bold")
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("State / Province")
    ax.invert_yaxis()
    ax.legend(title="Efficiency Score", bbox_to_anchor=(1.01, 1))
    save_chart("05_state_route_efficiency.png")


# ════════════════════════════════════════════════════════════════
# STEP 19 — CHART: HEATMAP — STATE vs DELAY STATUS
# ════════════════════════════════════════════════════════════════

def step19_chart_heatmap_delay(data: pd.DataFrame,
                                summary: pd.DataFrame) -> None:
    """
    Heatmap of Delay Status percentage for the top 25 states
    (by shipment volume).
    """
    print("\n" + "=" * 60)
    print("STEP 19: Chart — Heatmap State × Delay Status")
    print("=" * 60)

    if "Delay Status" not in data.columns:
        print("  Delay Status column not found — skipping.")
        return

    top25 = top_n_states(summary, "Total_Shipments", n=25)[STATE_COL].tolist()
    subset = data[data[STATE_COL].isin(top25)]

    pivot = (
        subset.groupby([STATE_COL, "Delay Status"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)
    pivot_pct = pivot_pct.reindex(top25)

    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(
        pivot_pct, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "% of Shipments"}, ax=ax,
    )
    ax.set_title("State × Delay Status Heatmap\n(Top 25 States by Shipment Volume)",
                 fontweight="bold")
    ax.set_xlabel("Delay Status")
    ax.set_ylabel("State / Province")
    save_chart("05_state_delay_heatmap.png")


# ════════════════════════════════════════════════════════════════
# STEP 20 — CHART: HEATMAP — STATE vs ROUTE EFFICIENCY
# ════════════════════════════════════════════════════════════════

def step20_chart_heatmap_efficiency(efficiency: pd.DataFrame,
                                     summary: pd.DataFrame) -> None:
    """
    Heatmap of Route Efficiency Score percentage for the top 25
    states by shipment volume.
    """
    print("\n" + "=" * 60)
    print("STEP 20: Chart — Heatmap State × Route Efficiency")
    print("=" * 60)

    if efficiency.empty:
        print("  Efficiency data not available — skipping.")
        return

    top25 = top_n_states(summary, "Total_Shipments", n=25)[STATE_COL].tolist()
    eff_cols = [c for c in EFF_SCORE_ORDER if c in efficiency.columns]
    subset = efficiency[efficiency[STATE_COL].isin(top25)].set_index(STATE_COL)
    subset = subset[eff_cols].reindex(top25)

    fig, ax = plt.subplots(figsize=(9, 14))
    sns.heatmap(
        subset, annot=True, fmt=".1f", cmap="RdYlGn",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "% of Shipments"}, ax=ax,
    )
    ax.set_title("State × Route Efficiency Heatmap\n(Top 25 States by Shipment Volume)",
                 fontweight="bold")
    ax.set_xlabel("Efficiency Score")
    ax.set_ylabel("State / Province")
    save_chart("05_state_efficiency_heatmap.png")


# ════════════════════════════════════════════════════════════════
# STEP 21 — CHART: BOX PLOT — LEAD TIME BY STATE (TOP 15)
# ════════════════════════════════════════════════════════════════

def step21_chart_boxplot_lead_time(data: pd.DataFrame,
                                    summary: pd.DataFrame) -> None:
    """
    Box plot of Shipping Lead Time (Simulated) distribution for the
    top 15 states by shipment count.
    """
    print("\n" + "=" * 60)
    print("STEP 21: Chart — Box Plot Lead Time by State")
    print("=" * 60)

    top15_states = top_n_states(summary, "Total_Shipments", n=15)[STATE_COL].tolist()
    subset = data[data[STATE_COL].isin(top15_states)]

    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(
        data=subset, x=STATE_COL, y=LEAD_TIME_COL,
        order=top15_states, palette=PALETTE_CAT, ax=ax,
    )
    ax.set_title("Lead Time Distribution by State (Top 15 by Volume)",
                 fontweight="bold")
    ax.set_xlabel("State / Province")
    ax.set_ylabel("Lead Time (days, Simulated)")
    ax.tick_params(axis="x", rotation=45)
    save_chart("05_state_leadtime_boxplot.png")


# ════════════════════════════════════════════════════════════════
# STEP 22 — CHART: SCATTER PLOT — SALES vs PROFIT
# ════════════════════════════════════════════════════════════════

def step22_chart_scatter(summary: pd.DataFrame) -> None:
    """Scatter plot of Total Sales vs Total Gross Profit per state."""
    print("\n" + "=" * 60)
    print("STEP 22: Chart — Scatter Plot Sales vs Profit")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette(PALETTE_CAT, len(summary))

    scatter = ax.scatter(
        summary["Total_Sales"], summary["Total_Profit"],
        c=range(len(summary)), cmap="Set2",
        s=80, alpha=0.75, edgecolors="white", linewidth=0.5,
    )

    # Label the top 10 states by sales
    top10 = top_n_states(summary, "Total_Sales", n=10)
    for _, row in top10.iterrows():
        ax.annotate(
            row[STATE_COL],
            (row["Total_Sales"], row["Total_Profit"]),
            fontsize=7, alpha=0.85,
            xytext=(4, 4), textcoords="offset points",
        )

    # Trend line
    x = summary["Total_Sales"]
    y = summary["Total_Profit"]
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color="red", linestyle="--",
            linewidth=1.5, label=f"Trend (r={x.corr(y):.2f})")

    ax.set_title("Total Sales vs Total Gross Profit by State",
                 fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    ax.set_ylabel("Total Gross Profit ($)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend()
    save_chart("05_state_scatter_sales_profit.png")


# ════════════════════════════════════════════════════════════════
# STEP 23 — CHART: HISTOGRAM — LEAD TIME DISTRIBUTION
# ════════════════════════════════════════════════════════════════

def step23_chart_histogram_leadtime(data: pd.DataFrame) -> None:
    """Histogram of the overall Shipping Lead Time (Simulated) distribution."""
    print("\n" + "=" * 60)
    print("STEP 23: Chart — Histogram Lead Time Distribution")
    print("=" * 60)

    series = pd.to_numeric(data[LEAD_TIME_COL], errors="coerce").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Shipping Lead Time (Simulated) — Distribution",
                 fontweight="bold", fontsize=14)

    axes[0].hist(series, bins=20, color=ACCENT_COLOR,
                 edgecolor="white", alpha=0.85)
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

    save_chart("05_state_leadtime_histogram.png")


# ════════════════════════════════════════════════════════════════
# STEP 24 — CHARTS: TOP/BOTTOM 10 STATES BY SALES & PROFIT
# ════════════════════════════════════════════════════════════════

def step24_chart_top_bottom(summary: pd.DataFrame) -> None:
    """
    Generate four charts:
      • Top 10 States by Sales
      • Bottom 10 States by Sales
      • Top 10 States by Profit
      • Bottom 10 States by Profit
    """
    print("\n" + "=" * 60)
    print("STEP 24: Charts — Top / Bottom 10 States")
    print("=" * 60)

    specs = [
        ("Total_Sales",   "Sales",  "Blues_r",  "Reds_r",
         "05_state_top10_sales.png",    "05_state_bottom10_sales.png"),
        ("Total_Profit",  "Profit", "Greens_r", "Oranges_r",
         "05_state_top10_profit.png",   "05_state_bottom10_profit.png"),
    ]

    for col, label, top_pal, bot_pal, top_file, bot_file in specs:
        # Top 10
        top10 = top_n_states(summary, col, n=10)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=top10, y=STATE_COL, x=col,
                    palette=top_pal, orient="h", ax=ax)
        ax.set_title(f"Top 10 States by {label}", fontweight="bold")
        ax.set_xlabel(f"Total {label} ($)")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        save_chart(top_file)

        # Bottom 10
        bot10 = top_n_states(summary, col, n=10, ascending=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=bot10, y=STATE_COL, x=col,
                    palette=bot_pal, orient="h", ax=ax)
        ax.set_title(f"Bottom 10 States by {label}", fontweight="bold")
        ax.set_xlabel(f"Total {label} ($)")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        save_chart(bot_file)


# ════════════════════════════════════════════════════════════════
# STEP 25 — CHARTS: DELAY DISTRIBUTION + PIE + REGION COMPARISON
# ════════════════════════════════════════════════════════════════

def step25_chart_delay_region(data: pd.DataFrame) -> None:
    """
    Three charts:
      • Delay Status overall distribution (count plot)
      • Pie chart: Shipment share by Region
      • Region Comparison bar chart (Sales, Profit, Shipments, Lead Time)
    """
    print("\n" + "=" * 60)
    print("STEP 25: Charts — Delay Distribution + Region")
    print("=" * 60)

    # ── (a) Delay Status Distribution ────────────────────────
    if "Delay Status" in data.columns:
        delay_counts = data["Delay Status"].value_counts()
        colors = [DELAY_STATUS_COLORS.get(c, "#95A5A6")
                  for c in delay_counts.index]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(delay_counts.index, delay_counts.values,
               color=colors, edgecolor="white")
        ax.set_title("Overall Delay Status Distribution", fontweight="bold")
        ax.set_xlabel("Delay Status")
        ax.set_ylabel("Order Count")
        for i, v in enumerate(delay_counts.values):
            ax.text(i, v + 20, f"{v:,}", ha="center", fontsize=10)
        save_chart("05_state_delay_distribution.png")

    # ── (b) Pie Chart — Shipment Share by Region ─────────────
    if "Region" in data.columns:
        region_counts = data["Region"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
            region_counts.values,
            labels=region_counts.index,
            autopct="%1.1f%%",
            colors=sns.color_palette(PALETTE_CAT, len(region_counts)),
            startangle=140,
        )
        ax.set_title("Shipment Share by Region", fontweight="bold")
        save_chart("05_state_region_pie.png")

    # ── (c) Region Comparison ────────────────────────────────
    if "Region" in data.columns:
        region_agg = (
            data.groupby("Region")
            .agg(
                Total_Sales=    ("Sales",       "sum"),
                Total_Profit=   ("Gross Profit","sum"),
                Shipments=      ("Order ID",    "count"),
                Avg_Lead_Time=  (LEAD_TIME_COL, "mean"),
            )
            .round(2)
            .reset_index()
        )
        metrics = ["Total_Sales", "Total_Profit", "Shipments", "Avg_Lead_Time"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Region Performance Comparison", fontweight="bold", fontsize=14)
        axes_flat = axes.flatten()
        region_colors = sns.color_palette(PALETTE_CAT, len(region_agg))

        for i, metric in enumerate(metrics):
            ax = axes_flat[i]
            ax.bar(region_agg["Region"], region_agg[metric],
                   color=region_colors, edgecolor="white")
            ax.set_title(metric.replace("_", " "), fontsize=11)
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=15)
            for bar in ax.patches:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f"{h:,.1f}", ha="center", va="bottom", fontsize=9)

        save_chart("05_state_region_comparison.png")


# ════════════════════════════════════════════════════════════════
# STEP 26 — CHART: OVERALL KPI COMPARISON (RADAR / MULTI-METRIC)
# ════════════════════════════════════════════════════════════════

def step26_chart_kpi_overview(summary: pd.DataFrame,
                               ranking: pd.DataFrame) -> None:
    """
    Multi-metric bar chart comparing the Top 10 overall ranked states
    across Sales, Profit, Lead Time and Delay Rate.
    """
    print("\n" + "=" * 60)
    print("STEP 26: Chart — Overall KPI Comparison")
    print("=" * 60)

    if ranking.empty:
        print("  Ranking data not available — skipping.")
        return

    top10_ranked = ranking.head(10)[STATE_COL].tolist()
    subset = summary[summary[STATE_COL].isin(top10_ranked)].copy()
    subset["Rank"] = subset[STATE_COL].map(
        {s: i + 1 for i, s in enumerate(top10_ranked)}
    )
    subset = subset.sort_values("Rank").reset_index(drop=True)

    metrics = {
        "Total_Sales":    ("Total Sales ($)",  ACCENT_COLOR),
        "Total_Profit":   ("Total Profit ($)", GREEN_COLOR),
        "Avg_Lead_Time":  ("Avg Lead Time (d)", ORANGE_COLOR),
        "Delay_Rate_%":   ("Delay Rate (%)",   RED_COLOR),
    }
    metrics = {k: v for k, v in metrics.items() if k in subset.columns}

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 7))
    fig.suptitle("KPI Comparison — Top 10 Overall Ranked States",
                 fontweight="bold", fontsize=14)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, (label, color)) in zip(axes, metrics.items()):
        ax.barh(subset[STATE_COL], subset[col], color=color, edgecolor="white")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(label)
        ax.invert_yaxis()

    save_chart("05_state_kpi_overview.png")


# ════════════════════════════════════════════════════════════════
# STEP 27 — BUSINESS INSIGHTS ENGINE
# ════════════════════════════════════════════════════════════════

def step27_business_insights(summary: pd.DataFrame,
                               ranking: pd.DataFrame) -> None:
    """
    Auto-generate at least 15 business insights from the state-level
    KPI summary and ranking tables.
    """
    print("\n" + "=" * 60)
    print("STEP 27: Business Insights")
    print("=" * 60)

    try:
        # ── Sales insights ────────────────────────────────────
        best_sales  = summary.loc[summary["Total_Sales"].idxmax()]
        worst_sales = summary.loc[summary["Total_Sales"].idxmin()]
        add_insight(
            f"Highest Sales State: {best_sales[STATE_COL]} "
            f"({fmt_currency(best_sales['Total_Sales'])})"
        )
        add_insight(
            f"Lowest Sales State: {worst_sales[STATE_COL]} "
            f"({fmt_currency(worst_sales['Total_Sales'])})"
        )

        # ── Profit insights ───────────────────────────────────
        best_profit  = summary.loc[summary["Total_Profit"].idxmax()]
        worst_profit = summary.loc[summary["Total_Profit"].idxmin()]
        add_insight(
            f"Highest Profit State: {best_profit[STATE_COL]} "
            f"({fmt_currency(best_profit['Total_Profit'])})"
        )
        add_insight(
            f"Lowest Profit State: {worst_profit[STATE_COL]} "
            f"({fmt_currency(worst_profit['Total_Profit'])})"
        )

        # ── Margin insight ────────────────────────────────────
        if "Avg_Profit_Margin" in summary.columns:
            best_margin = summary.loc[summary["Avg_Profit_Margin"].idxmax()]
            add_insight(
                f"Best Profit Margin State: {best_margin[STATE_COL]} "
                f"({fmt_pct(best_margin['Avg_Profit_Margin'])} avg margin)"
            )

        # ── Lead Time insights ────────────────────────────────
        fastest = summary.loc[summary["Avg_Lead_Time"].idxmin()]
        slowest = summary.loc[summary["Avg_Lead_Time"].idxmax()]
        add_insight(
            f"Fastest Shipping State: {fastest[STATE_COL]} "
            f"({fastest['Avg_Lead_Time']:.2f} days avg lead time)"
        )
        add_insight(
            f"Slowest Shipping State: {slowest[STATE_COL]} "
            f"({slowest['Avg_Lead_Time']:.2f} days avg lead time)"
        )

        # ── Delay insights ────────────────────────────────────
        if "Delay_Rate_%" in summary.columns:
            highest_delay = summary.loc[summary["Delay_Rate_%"].idxmax()]
            lowest_delay  = summary.loc[summary["Delay_Rate_%"].idxmin()]
            add_insight(
                f"Highest Delay State: {highest_delay[STATE_COL]} "
                f"({fmt_pct(highest_delay['Delay_Rate_%'])} delayed)"
            )
            add_insight(
                f"Lowest Delay State: {lowest_delay[STATE_COL]} "
                f"({fmt_pct(lowest_delay['Delay_Rate_%'])} delayed)"
            )

        # ── Volume insight ────────────────────────────────────
        most_shipments = summary.loc[summary["Total_Shipments"].idxmax()]
        total_ship     = summary["Total_Shipments"].sum()
        add_insight(
            f"Most Shipments State: {most_shipments[STATE_COL]} "
            f"({int(most_shipments['Total_Shipments']):,} orders = "
            f"{most_shipments['Total_Shipments'] / total_ship * 100:.1f}% of all)"
        )

        # ── Route Efficiency insights ─────────────────────────
        if not ranking.empty and "Efficiency Rank" in ranking.columns:
            best_eff  = ranking.loc[ranking["Efficiency Rank"].idxmin()]
            worst_eff = ranking.loc[ranking["Efficiency Rank"].idxmax()]
            add_insight(
                f"Highest Route Efficiency State: {best_eff[STATE_COL]}"
            )
            add_insight(
                f"Lowest Route Efficiency State: {worst_eff[STATE_COL]}"
            )

        # ── Improvement opportunity ───────────────────────────
        if "Delay_Rate_%" in summary.columns and "Total_Sales" in summary.columns:
            opportunity = summary[summary["Delay_Rate_%"] > 30].sort_values(
                "Total_Sales", ascending=False
            )
            if not opportunity.empty:
                top_opp = opportunity.iloc[0]
                add_insight(
                    f"Top Improvement Opportunity: {top_opp[STATE_COL]} — "
                    f"high sales ({fmt_currency(top_opp['Total_Sales'])}) but "
                    f"elevated delay rate ({fmt_pct(top_opp['Delay_Rate_%'])})"
                )

        # ── Financial overview ─────────────────────────────────
        add_insight(
            f"Combined Total Revenue (all states): "
            f"{fmt_currency(summary['Total_Sales'].sum())}"
        )
        add_insight(
            f"Combined Total Gross Profit (all states): "
            f"{fmt_currency(summary['Total_Profit'].sum())}"
        )
        add_insight(
            f"Overall Average Lead Time (all states): "
            f"{summary['Avg_Lead_Time'].mean():.2f} days"
        )

    except Exception as exc:
        print(f"  WARNING: Insight generation encountered an issue: {exc}")

    insights_df = pd.DataFrame({
        "Insight #": range(1, len(insight_list) + 1),
        "Insight":   insight_list,
    })
    save_summary(insights_df, "state_business_insights.csv")
    print(f"\n  Total business insights generated: {len(insight_list)}")


# ════════════════════════════════════════════════════════════════
# STEP 28 — RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════

def step28_recommendation_engine(summary: pd.DataFrame,
                                   ranking: pd.DataFrame) -> str:
    """
    Auto-generate data-driven business recommendations covering:
    inventory allocation, carrier optimization, delay reduction,
    warehouse planning, route optimization, cost reduction,
    regional expansion, and resource allocation.
    """
    print("\n" + "=" * 60)
    print("STEP 28: Recommendation Engine")
    print("=" * 60)

    recs = []

    try:
        # Inventory Allocation
        top3_sales = summary.nlargest(3, "Total_Sales")[STATE_COL].tolist()
        recs.append(
            "1. INVENTORY ALLOCATION\n"
            f"   Prioritize safety stock in {', '.join(top3_sales)} — these states "
            "generate the highest cumulative sales. Aligning warehouse buffer levels "
            "with seasonal demand peaks in these markets will reduce stockout risk."
        )

        # Carrier Optimization
        if "Avg_Lead_Time" in summary.columns:
            slow_states = summary.nlargest(5, "Avg_Lead_Time")[STATE_COL].tolist()
            recs.append(
                "2. CARRIER OPTIMIZATION\n"
                f"   Renegotiate carrier SLAs or introduce alternative carriers for "
                f"{', '.join(slow_states)} — these states consistently record the "
                "longest average shipping lead times and are prime candidates for "
                "expedited or dedicated lane agreements."
            )

        # Delay Reduction
        if "Delay_Rate_%" in summary.columns:
            high_delay = summary[summary["Delay_Rate_%"] > 35].nlargest(
                5, "Delay_Rate_%"
            )[STATE_COL].tolist()
            if high_delay:
                recs.append(
                    "3. DELAY REDUCTION\n"
                    f"   States with delay rates above 35%: {', '.join(high_delay)}. "
                    "Introduce real-time shipment tracking, buffer days for Standard Class "
                    "routes into these states, and set up proactive exception alerts for "
                    "orders approaching the SLA breach threshold."
                )
            else:
                recs.append(
                    "3. DELAY REDUCTION\n"
                    "   No state currently exceeds a 35% delay rate — maintain existing "
                    "SLA discipline and conduct quarterly carrier performance reviews."
                )

        # Warehouse Planning
        top5_vol = summary.nlargest(5, "Total_Shipments")[STATE_COL].tolist()
        recs.append(
            "4. WAREHOUSE PLANNING\n"
            f"   Evaluate regional distribution center (DC) placement near "
            f"{', '.join(top5_vol)} to reduce last-mile lead times. A DC positioned "
            "centrally within the highest-volume cluster could simultaneously serve "
            "multiple top-10 states and reduce cross-country freight costs."
        )

        # Route Optimization
        if not ranking.empty and "Lead Time Rank" in ranking.columns:
            worst_lt = ranking.nlargest(5, "Lead Time Rank")[STATE_COL].tolist()
            recs.append(
                "5. ROUTE OPTIMIZATION\n"
                f"   Review factory-to-state routing for {', '.join(worst_lt)}. "
                "Shifting volume from Standard Class to Second Class on these routes, "
                "or introducing direct factory-to-DC consolidation lanes, may yield "
                "meaningful lead-time reductions without proportional cost increases."
            )

        # Cost Reduction
        if "Total_Cost" in summary.columns:
            high_cost = summary.nlargest(5, "Total_Cost")[STATE_COL].tolist()
            recs.append(
                "6. COST REDUCTION\n"
                f"   Total cost concentration is highest in {', '.join(high_cost)}. "
                "Conduct a cost-per-unit benchmark against comparable routes and "
                "identify consolidation opportunities (e.g., full-truckload vs. "
                "LTL) to lower per-shipment costs in these markets."
            )

        # Regional Expansion
        bot5_sales = summary.nsmallest(5, "Total_Sales")[STATE_COL].tolist()
        recs.append(
            "7. REGIONAL EXPANSION\n"
            f"   Low-penetration states: {', '.join(bot5_sales)}. Assess whether "
            "these markets represent genuinely low demand or are simply "
            "under-served by current distribution infrastructure. A targeted "
            "promotional campaign combined with reduced shipping thresholds could "
            "stimulate incremental volume."
        )

        # Resource Allocation
        if not ranking.empty:
            top5_ranked = ranking.head(5)[STATE_COL].tolist()
            recs.append(
                "8. RESOURCE ALLOCATION\n"
                f"   Direct field sales and customer success resources towards "
                f"{', '.join(top5_ranked)} — the top composite-ranked states that "
                "combine strong revenue, low delay rates and high route efficiency. "
                "Protecting and expanding share in best-performing markets delivers "
                "the highest return on resource investment."
            )

    except Exception as exc:
        print(f"  WARNING: Recommendation engine encountered an issue: {exc}")

    report_text = (
        "STATE-LEVEL BUSINESS RECOMMENDATIONS\n"
        "Factory-to-Customer Shipping Route Efficiency Analysis\n"
        "Nassau Candy Distributor\n"
        + "=" * 60 + "\n\n"
        + "\n\n".join(recs) + "\n"
    )

    save_report_text(report_text, "state_recommendations.txt")
    for rec in recs:
        print(f"\n  {rec}")

    return report_text


# ════════════════════════════════════════════════════════════════
# STEP 29 — EXECUTIVE REPORT
# ════════════════════════════════════════════════════════════════

def step29_executive_report(summary: pd.DataFrame,
                              ranking: pd.DataFrame,
                              anova_df: pd.DataFrame,
                              recommendations_text: str) -> None:
    """
    Generate a professional management-level executive report
    containing: Executive Summary, Key Findings, State Rankings,
    Statistical Findings, Business Insights, Recommendations and
    a Conclusion.
    """
    print("\n" + "=" * 60)
    print("STEP 29: Executive Report")
    print("=" * 60)

    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        best_sales  = summary.loc[summary["Total_Sales"].idxmax()]
        best_profit = summary.loc[summary["Total_Profit"].idxmax()]
        fastest     = summary.loc[summary["Avg_Lead_Time"].idxmin()]
        slowest     = summary.loc[summary["Avg_Lead_Time"].idxmax()]

        lowest_delay_state  = "N/A"
        highest_delay_state = "N/A"
        if "Delay_Rate_%" in summary.columns:
            lowest_delay_state  = summary.loc[
                summary["Delay_Rate_%"].idxmin(), STATE_COL]
            highest_delay_state = summary.loc[
                summary["Delay_Rate_%"].idxmax(), STATE_COL]

        lines = [
            "=" * 60,
            "EXECUTIVE REPORT — STATE-LEVEL SHIPPING PERFORMANCE",
            "Factory-to-Customer Shipping Route Efficiency Analysis",
            "Nassau Candy Distributor",
            f"Generated: {ts}",
            "=" * 60,
            "",
            section("EXECUTIVE SUMMARY").strip(),
            (
                f"This report presents a comprehensive state-level analysis of "
                f"shipping performance across {len(summary)} states and provinces "
                f"served by Nassau Candy Distributor. The analysis covers total sales, "
                f"gross profit, shipping lead time, delay rates and route efficiency, "
                f"and concludes with data-driven business recommendations."
            ),
            "",
            section("KEY FINDINGS").strip(),
            f"  • Highest Sales State       : {best_sales[STATE_COL]} "
            f"({fmt_currency(best_sales['Total_Sales'])})",
            f"  • Highest Profit State      : {best_profit[STATE_COL]} "
            f"({fmt_currency(best_profit['Total_Profit'])})",
            f"  • Fastest Shipping State    : {fastest[STATE_COL]} "
            f"({fastest['Avg_Lead_Time']:.2f} days avg)",
            f"  • Slowest Shipping State    : {slowest[STATE_COL]} "
            f"({slowest['Avg_Lead_Time']:.2f} days avg)",
            f"  • Lowest Delay State        : {lowest_delay_state}",
            f"  • Highest Delay State       : {highest_delay_state}",
            f"  • Total Revenue (all states): "
            f"{fmt_currency(summary['Total_Sales'].sum())}",
            f"  • Total Gross Profit        : "
            f"{fmt_currency(summary['Total_Profit'].sum())}",
            f"  • Overall Avg Lead Time     : "
            f"{summary['Avg_Lead_Time'].mean():.2f} days",
        ]

        if not ranking.empty:
            lines += [
                "",
                section("TOP 10 OVERALL STATE RANKINGS").strip(),
            ]
            for _, row in ranking.head(10).iterrows():
                lines.append(
                    f"  Rank {int(row['Overall Rank']):>2}: {row[STATE_COL]}"
                )

        if not anova_df.empty:
            lines += [
                "",
                section("STATISTICAL SIGNIFICANCE (ANOVA)").strip(),
            ]
            for _, row in anova_df.iterrows():
                lines.append(
                    f"  • {row['Metric']}: F={row['F-Statistic']}, "
                    f"p={row['P-Value']} "
                    f"(State effect significant: {row['Significant (p<0.05)']})"
                )

        lines += [
            "",
            section("BUSINESS INSIGHTS").strip(),
        ]
        for insight in insight_list:
            lines.append(f"  • {insight}")

        lines += [
            "",
            section("RECOMMENDATIONS").strip(),
            recommendations_text.split("=" * 60, 1)[-1].strip(),
            "",
            section("CONCLUSION").strip(),
            textwrap.fill(
                "State-level shipping performance varies significantly across "
                "Nassau Candy Distributor's geographic footprint. By concentrating "
                "resources in high-performing markets, reducing delay rates in "
                "identified hotspot states, and optimizing carrier and route "
                "assignments, the company can materially improve customer "
                "satisfaction, reduce logistics costs, and accelerate revenue "
                "growth across all regions.",
                width=68,
            ),
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ]

        report_text = "\n".join(lines) + "\n"
        save_report_text(report_text, "state_executive_report.txt")

    except Exception as exc:
        print(f"  WARNING: Executive report generation failed: {exc}")


# ════════════════════════════════════════════════════════════════
# STEP 30 — DASHBOARD-READY DATASETS
# ════════════════════════════════════════════════════════════════

def step30_dashboard_datasets(data: pd.DataFrame,
                                kpi: pd.DataFrame) -> None:
    """
    Build two lightweight datasets for the Streamlit dashboard's
    State Analysis module:
      dashboard_state.csv     — row-level data with key columns only
      dashboard_state_kpi.csv — aggregated KPI table per state
    """
    print("\n" + "=" * 60)
    print("STEP 30: Dashboard-Ready Datasets")
    print("=" * 60)

    dashboard_cols = [
        "Order ID", STATE_COL, "Region", "City", "Order Date",
        "Factory", "Ship Mode", "Sales", "Gross Profit",
        "Profit Margin %", LEAD_TIME_COL, "Delay Status",
        "Route Efficiency Score", "Units", "Cost",
    ]
    dashboard_cols = [c for c in dashboard_cols if c in data.columns]

    try:
        dash_df = data[dashboard_cols].copy()
        dash_df = dash_df.dropna(subset=[STATE_COL])
        save_summary(dash_df,  "dashboard_state.csv")

        if not kpi.empty:
            save_summary(kpi, "dashboard_state_kpi.csv")

        print(f"  Row-level dataset shape : {dash_df.shape}")
        print(f"  KPI dataset shape       : {kpi.shape if not kpi.empty else 'N/A'}")
        print(f"  Columns in row dataset  : {dashboard_cols}")

    except Exception as exc:
        print(f"  WARNING: Dashboard dataset generation failed: {exc}")


# ════════════════════════════════════════════════════════════════
# STEP 31 — FINAL EXECUTION REPORT
# ════════════════════════════════════════════════════════════════

def step31_final_report() -> None:
    """
    Print a complete execution summary showing total charts, summary
    tables, text reports and business insights generated.
    """
    print("\n" + "=" * 60)
    print("STEP 31: Final Execution Report")
    print("=" * 60)
    print(f"  Total Charts Generated      : {chart_count}")
    print(f"  Total Summary Tables Saved  : {summary_count}")
    print(f"  Total Text Reports Saved    : {report_count}")
    print(f"  Total Business Insights     : {len(insight_list)}")
    print(f"  Charts saved in             : ./{CHARTS_DIR}/")
    print(f"  Summaries & reports saved in: ./{SUMMARIES_DIR}/")
    print("  State-Level Analysis Completed Successfully")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main() -> None:
    """Execute the complete state-level analysis pipeline."""
    try:
        # ── Data loading & quality ────────────────────────────
        df = step1_load_dataset()
        step2_data_quality(df)

        # ── Core aggregations ─────────────────────────────────
        summary    = step3_build_state_summary(df)
        efficiency = step4_route_efficiency_by_state(df)
        comparison = step5_state_comparison(summary, efficiency)
        delay_df   = step6_delay_summary(df)

        # ── Statistical analysis ──────────────────────────────
        step7_statistical_summary(df)
        step8_outlier_detection(df)
        step9_correlation_analysis(df)
        anova_df = step10_anova_test(df)

        # ── Ranking & KPI ─────────────────────────────────────
        ranking = step11_state_ranking(summary, efficiency)
        kpi     = step12_kpi_summary(summary, efficiency, ranking)

        # ── Visualizations ────────────────────────────────────
        step13_chart_sales_by_state(summary)
        step14_chart_profit_by_state(summary)
        step15_chart_lead_time(summary)
        step16_chart_delay_rate(summary)
        step17_chart_shipment_count(summary)
        step18_chart_route_efficiency(efficiency)
        step19_chart_heatmap_delay(df, summary)
        step20_chart_heatmap_efficiency(efficiency, summary)
        step21_chart_boxplot_lead_time(df, summary)
        step22_chart_scatter(summary)
        step23_chart_histogram_leadtime(df)
        step24_chart_top_bottom(summary)
        step25_chart_delay_region(df)
        step26_chart_kpi_overview(summary, ranking)

        # ── Insights, recommendations & reports ───────────────
        step27_business_insights(summary, ranking)
        recs_text = step28_recommendation_engine(summary, ranking)
        step29_executive_report(summary, ranking, anova_df, recs_text)

        # ── Dashboard datasets & final report ─────────────────
        step30_dashboard_datasets(df, kpi)
        step31_final_report()

    except Exception as exc:
        print(f"\nPIPELINE FAILED: {exc}")
        raise


if __name__ == "__main__":
    main()