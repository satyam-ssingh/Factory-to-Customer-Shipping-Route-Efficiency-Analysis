"""
03_exploratory_data_analysis.py
---------------------------------
Phase 3 - Exploratory Data Analysis (EDA)
Project: Factory-to-Customer Shipping Route Efficiency Analysis
         for Nassau Candy Distributor

Input  : featured_nassau_candy.csv  (output of 02_feature_engineering.py)
Output : EDA_Charts/   (all high-resolution visualizations)
         EDA_Summaries/ (all summary CSVs)

Run after 01_data_cleaning.py and 02_feature_engineering.py.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────────
CHARTS_DIR    = "EDA_Charts"
SUMMARIES_DIR = "EDA_Summaries"
os.makedirs(CHARTS_DIR,    exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# COUNTERS
# ─────────────────────────────────────────────
chart_count   = 0
summary_count = 0
insight_list  = []


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def save_chart(filename: str) -> None:
    """Save current figure to EDA_Charts/ and close it."""
    global chart_count
    path = os.path.join(CHARTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    chart_count += 1
    print(f"  [Chart saved] {path}")


def save_summary(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to EDA_Summaries/ as CSV."""
    global summary_count
    path = os.path.join(SUMMARIES_DIR, filename)
    df.to_csv(path, index=False, float_format="%.2f")
    summary_count += 1
    print(f"  [Summary saved] {path}")


def add_insight(text: str) -> None:
    """Append a business insight to the global list."""
    insight_list.append(text)
    print(f"  [Insight {len(insight_list):02d}] {text}")


def fmt_currency(val: float) -> str:
    """Format a float as $X,XXX."""
    return f"${val:,.2f}"


def add_value_labels(ax, fmt="{:.1f}", fontsize=9, color="black"):
    """Add value labels on top of each bar in a bar chart."""
    for p in ax.patches:
        height = p.get_height()
        if height == 0 or np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", va="bottom",
            fontsize=fontsize, color=color,
        )


# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATASET
# ════════════════════════════════════════════════════════════════

def step1_load_dataset(path: str = "featured_nassau_candy.csv") -> pd.DataFrame:
    """Load featured dataset and display structural info."""
    print("\n" + "=" * 60)
    print("STEP 1: Load Dataset")
    print("=" * 60)

    try:
        df = pd.read_csv(
            path,
            parse_dates=["Order Date", "Simulated Ship Date"],
            low_memory=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{path}' not found. Run 01_data_cleaning.py and "
            "02_feature_engineering.py first."
        )

    print(f"Dataset Shape       : {df.shape}")
    print(f"Number of Rows      : {df.shape[0]:,}")
    print(f"Number of Columns   : {df.shape[1]}")
    print("\nColumn Names:")
    for col in df.columns:
        print(f"  {col}")
    print("\nData Types:")
    print(df.dtypes.to_string())
    print("\nFirst Five Rows:")
    print(df.head().to_string())

    return df


# ════════════════════════════════════════════════════════════════
# STEP 2 — DATA QUALITY CHECK
# ════════════════════════════════════════════════════════════════

def step2_data_quality(df: pd.DataFrame) -> None:
    """Check missing values, duplicates, summary statistics, unique categoricals."""
    print("\n" + "=" * 60)
    print("STEP 2: Data Quality Check")
    print("=" * 60)

    # Missing values (ignore unused original columns)
    ignore_cols = ["Ship Date", "Shipping Lead Time"]

    missing = df.drop(columns=ignore_cols, errors="ignore").isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })

    missing_df = missing_df[missing_df["Missing Count"] > 0]
    print("\nMissing Values:")
    if missing_df.empty:
        print("  None — dataset is complete.")
    else:
        print(missing_df.to_string())

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dup_count}")

    # Summary statistics
    numeric_df = df.drop(columns=["Ship Date", "Shipping Lead Time"], errors="ignore")

    numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        print("\nSummary Statistics (Numeric):")
        print(numeric_df[numeric_cols].describe().round(2).to_string())
    else:
        print("\nNo numeric columns found.")

    # Unique values for categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("\nUnique Values per Categorical Column:")
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values")


# ════════════════════════════════════════════════════════════════
# STEP 3 — UNIVARIATE ANALYSIS
# ════════════════════════════════════════════════════════════════

def _dist_stats(series: pd.Series, name: str) -> dict:
    """Return distribution statistics for a numeric series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = series[(series < lower_fence) | (series > upper_fence)]
    return {
        "Column": name,
        "Mean": round(series.mean(), 2),
        "Median": round(series.median(), 2),
        "Mode": round(series.mode().iloc[0], 2) if not series.mode().empty else np.nan,
        "Std Dev": round(series.std(), 2),
        "Min": round(series.min(), 2),
        "Max": round(series.max(), 2),
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "IQR": round(iqr, 2),
        "Outlier Count": len(outliers),
    }


def step3_univariate(df: pd.DataFrame) -> None:
    """Histogram, box plot and stats for each key numeric column."""
    print("\n" + "=" * 60)
    print("STEP 3: Univariate Analysis")
    print("=" * 60)

    num_cols = [
        "Sales", "Units", "Cost", "Gross Profit",
        "Profit Margin %", "Shipping Lead Time (Simulated)",
    ]
    # Keep only columns that actually exist
    num_cols = [c for c in num_cols if c in df.columns]

    stats_rows = []

    for col in num_cols:
        series = df[col].dropna()
        st = _dist_stats(series, col)
        stats_rows.append(st)
        print(f"\n  {col}:")
        for k, v in st.items():
            if k != "Column":
                print(f"    {k:20s}: {v}")

        # ── Histogram ──────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Distribution of {col}", fontsize=15, fontweight="bold")

        axes[0].hist(series, bins=40, color=ACCENT_COLOR, edgecolor="white", alpha=0.85)
        axes[0].set_title("Histogram")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        # ── Box Plot ──────────────────────────────────────────
        axes[1].boxplot(series, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=ACCENT_COLOR, alpha=0.6),
                        medianprops=dict(color="red", linewidth=2))
        axes[1].set_title("Box Plot")
        axes[1].set_ylabel(col)

        save_chart(f"03_univariate_{col.replace(' ', '_').replace('%', 'pct')}.png")

    stats_df = pd.DataFrame(stats_rows)
    save_summary(stats_df, "univariate_stats.csv")


# ════════════════════════════════════════════════════════════════
# STEP 4 — CATEGORICAL ANALYSIS
# ════════════════════════════════════════════════════════════════

def step4_categorical(df: pd.DataFrame) -> None:
    """Count plots, bar charts, pie charts and frequency tables for categoricals."""
    print("\n" + "=" * 60)
    print("STEP 4: Categorical Analysis")
    print("=" * 60)

    cat_targets = [
        "Factory", "Division", "Region", "State/Province",
        "Ship Mode", "Delay Status", "Route Efficiency Score", "Product Name",
    ]
    cat_targets = [c for c in cat_targets if c in df.columns]

    for col in cat_targets:
        freq = df[col].value_counts().reset_index()
        freq.columns = [col, "Count"]
        freq["Percentage"] = (freq["Count"] / len(df) * 100).round(2)
        save_summary(freq, f"freq_{col.replace('/', '_').replace(' ', '_')}.csv")
        print(f"\n  {col}:")
        print(freq.to_string(index=False))

        # Count plot
        top_n = freq.head(15)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Analysis: {col}", fontsize=14, fontweight="bold")

        sns.barplot(data=top_n, x="Count", y=col, palette=PALETTE_CAT,
                    ax=axes[0], orient="h")
        axes[0].set_title("Count Distribution")
        axes[0].set_xlabel("Count")
        axes[0].set_ylabel(col)

        # Pie chart (only for columns with ≤ 10 unique values)
        if freq.shape[0] <= 10:
            axes[1].pie(
                freq["Count"],
                labels=freq[col],
                autopct="%1.1f%%",
                colors=sns.color_palette(PALETTE_CAT, len(freq)),
                startangle=140,
            )
            axes[1].set_title("Proportion (%)")
        else:
            # Replace pie with a second bar chart for high-cardinality columns
            top15 = freq.head(15)
            sns.barplot(data=top15, x=col, y="Count",
                        palette=PALETTE_CAT, ax=axes[1])
            axes[1].set_title("Top 15 by Count")
            axes[1].tick_params(axis="x", rotation=45)

        save_chart(f"04_categorical_{col.replace('/', '_').replace(' ', '_')}.png")


# ════════════════════════════════════════════════════════════════
# STEP 5 — SALES ANALYSIS
# ════════════════════════════════════════════════════════════════

def step5_sales(df: pd.DataFrame) -> None:
    """Comprehensive sales analysis across products, factories, states, regions."""
    print("\n" + "=" * 60)
    print("STEP 5: Sales Analysis")
    print("=" * 60)

    total_sales  = df["Sales"].sum()
    avg_sales    = df["Sales"].mean()
    median_sales = df["Sales"].median()
    print(f"  Total Sales   : {fmt_currency(total_sales)}")
    print(f"  Average Sales : {fmt_currency(avg_sales)}")
    print(f"  Median Sales  : {fmt_currency(median_sales)}")

    # ── Sales Distribution ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df["Sales"], bins=50, color=ACCENT_COLOR, edgecolor="white", alpha=0.85)
    ax.set_title("Sales Distribution", fontweight="bold")
    ax.set_xlabel("Sales ($)")
    ax.set_ylabel("Frequency")
    save_chart("05_sales_distribution.png")

    # Aggregation helper
    def agg_sales(group_col: str, top: int = 10) -> pd.DataFrame:
        return (
            df.groupby(group_col)["Sales"]
            .agg(Total_Sales="sum", Avg_Sales="mean", Order_Count="count")
            .round(2)
            .sort_values("Total_Sales", ascending=False)
            .head(top)
            .reset_index()
        )

    dims = {
        "Product Name": ("Top 10 Products by Sales", "product"),
        "Factory":      ("Top Factories by Sales",   "factory"),
        "State/Province": ("Top 10 States by Sales", "state"),
        "Region":       ("Top Regions by Sales",     "region"),
        "Ship Mode":    ("Top Ship Modes by Sales",  "shipmode"),
    }

    all_sales_rows = []
    for col, (title, tag) in dims.items():
        if col not in df.columns:
            continue
        agg = agg_sales(col, top=10 if col not in ("Factory", "Region", "Ship Mode") else 20)
        save_summary(agg, f"sales_{tag}.csv")
        all_sales_rows.append(agg.assign(Dimension=col))

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=agg, y=col, x="Total_Sales",
                    palette=PALETTE_SEQ, ax=ax, orient="h")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Total Sales ($)")
        ax.set_ylabel(col)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        save_chart(f"05_sales_{tag}.png")
        print(f"\n  {title}:")
        print(agg.to_string(index=False))

    # Bottom 10 products
    bottom10 = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values()
        .head(10)
        .reset_index()
    )
    bottom10.columns = ["Product Name", "Total_Sales"]
    save_summary(bottom10, "sales_bottom10_products.csv")

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=bottom10, y="Product Name", x="Total_Sales",
                palette="Reds_r", ax=ax, orient="h")
    ax.set_title("Bottom 10 Products by Sales", fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    save_chart("05_sales_bottom10_products.png")

    save_summary(
        pd.DataFrame({
            "Metric": ["Total Sales", "Average Sales", "Median Sales"],
            "Value":  [total_sales, avg_sales, median_sales],
        }),
        "sales_summary.csv",
    )


# ════════════════════════════════════════════════════════════════
# STEP 6 — PROFIT ANALYSIS
# ════════════════════════════════════════════════════════════════

def step6_profit(df: pd.DataFrame) -> None:
    """Gross profit and profit margin analysis."""
    print("\n" + "=" * 60)
    print("STEP 6: Profit Analysis")
    print("=" * 60)

    total_profit = df["Gross Profit"].sum()
    avg_profit   = df["Gross Profit"].mean()
    avg_margin   = df["Profit Margin %"].mean() if "Profit Margin %" in df.columns else np.nan
    print(f"  Total Gross Profit  : {fmt_currency(total_profit)}")
    print(f"  Average Profit      : {fmt_currency(avg_profit)}")
    print(f"  Average Margin %    : {avg_margin:.2f}%")

    dims = {
        "Product Name":   ("Top 10 Products by Profit",   "product"),
        "Factory":        ("Factory Profit",               "factory"),
        "State/Province": ("Top 10 States by Profit",     "state"),
        "Region":         ("Region Profit",                "region"),
        "Ship Mode":      ("Ship Mode Profit",             "shipmode"),
    }

    profit_rows = []
    for col, (title, tag) in dims.items():
        if col not in df.columns:
            continue
        agg = (
            df.groupby(col)
            .agg(
                Total_Profit=("Gross Profit", "sum"),
                Avg_Margin=("Profit Margin %", "mean"),
                Orders=("Sales", "count"),
            )
            .round(2)
            .sort_values("Total_Profit", ascending=False)
            .head(10)
            .reset_index()
        )
        save_summary(agg, f"profit_{tag}.csv")
        profit_rows.append(agg.assign(Dimension=col))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        sns.barplot(data=agg, y=col, x="Total_Profit",
                    palette="Greens_r", ax=axes[0], orient="h")
        axes[0].set_title("Total Gross Profit")
        axes[0].set_xlabel("Gross Profit ($)")

        sns.barplot(data=agg, y=col, x="Avg_Margin",
                    palette="YlGn", ax=axes[1], orient="h")
        axes[1].set_title("Average Profit Margin %")
        axes[1].set_xlabel("Profit Margin %")

        save_chart(f"06_profit_{tag}.png")
        print(f"\n  {title}:")
        print(agg.to_string(index=False))

    # Bottom 10 products by profit
    bottom10_profit = (
        df.groupby("Product Name")["Gross Profit"]
        .sum()
        .sort_values()
        .head(10)
        .reset_index()
    )
    bottom10_profit.columns = ["Product Name", "Total_Profit"]
    save_summary(bottom10_profit, "profit_bottom10_products.csv")

    save_summary(
        pd.DataFrame({
            "Metric": ["Total Gross Profit", "Average Profit", "Average Margin %"],
            "Value":  [total_profit, avg_profit, avg_margin],
        }),
        "profit_summary.csv",
    )


# ════════════════════════════════════════════════════════════════
# STEP 7 — SHIPPING LEAD TIME ANALYSIS
# ════════════════════════════════════════════════════════════════

def step7_lead_time(df: pd.DataFrame) -> None:
    """Lead time distribution and breakdown by factory/region/state/ship mode/product."""
    print("\n" + "=" * 60)
    print("STEP 7: Shipping Lead Time Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"
    if lt_col not in df.columns:
        print("  Lead time column not found — skipping.")
        return

    series = df[lt_col].dropna()
    print(f"  Average Lead Time : {series.mean():.2f} days")
    print(f"  Median Lead Time  : {series.median():.2f} days")
    print(f"  Max Lead Time     : {series.max():.2f} days")
    print(f"  Min Lead Time     : {series.min():.2f} days")

    # Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Shipping Lead Time (Simulated) Distribution", fontweight="bold")
    axes[0].hist(series, bins=20, color=ACCENT_COLOR, edgecolor="white", alpha=0.85)
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Lead Time (days)")
    axes[0].set_ylabel("Frequency")
    axes[1].boxplot(series, patch_artist=True,
                    boxprops=dict(facecolor=ACCENT_COLOR, alpha=0.6),
                    medianprops=dict(color="red", linewidth=2))
    axes[1].set_title("Box Plot")
    axes[1].set_ylabel("Lead Time (days)")
    save_chart("07_lead_time_distribution.png")

    dims = ["Factory", "Region", "State/Province", "Ship Mode", "Product Name"]
    for col in dims:
        if col not in df.columns:
            continue
        agg = (
            df.groupby(col)[lt_col]
            .agg(Avg_Lead_Time="mean", Median_Lead_Time="median",
                 Max_Lead_Time="max", Min_Lead_Time="min", Count="count")
            .round(2)
            .sort_values("Avg_Lead_Time", ascending=False)
            .reset_index()
        )
        save_summary(agg, f"lead_time_{col.replace('/', '_').replace(' ', '_')}.csv")

        top_n = agg.head(15)
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=top_n, y=col, x="Avg_Lead_Time",
                    palette="coolwarm", ax=ax, orient="h")
        ax.set_title(f"Average Lead Time by {col}", fontweight="bold")
        ax.set_xlabel("Avg Lead Time (days)")
        ax.axvline(series.mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"Overall avg: {series.mean():.1f}d")
        ax.legend()
        save_chart(f"07_lead_time_by_{col.replace('/', '_').replace(' ', '_')}.png")
        print(f"\n  Lead Time by {col}:")
        print(agg.head(10).to_string(index=False))


# ════════════════════════════════════════════════════════════════
# STEP 8 — FACTORY PERFORMANCE ANALYSIS
# ════════════════════════════════════════════════════════════════

def step8_factory_performance(df: pd.DataFrame) -> None:
    """Full factory-level KPIs."""
    print("\n" + "=" * 60)
    print("STEP 8: Factory Performance Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"
    agg = (
        df.groupby("Factory")
        .agg(
            Shipments=("Order ID", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Avg_Lead_Time=(lt_col, "mean"),
            Avg_Profit_Margin=("Profit Margin %", "mean"),
        )
        .round(2)
        .sort_values("Shipments", ascending=False)
        .reset_index()
    )

    if "Route Efficiency Score" in df.columns:
        eff = (
            df.groupby("Factory")["Route Efficiency Score"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .rename("Pct")
            .reset_index()
        )
        pivot_eff = eff.pivot(index="Factory",
                               columns="Route Efficiency Score",
                               values="Pct").fillna(0).reset_index()
        agg = agg.merge(pivot_eff, on="Factory", how="left")

    save_summary(agg, "factory_summary.csv")
    print(agg.to_string(index=False))

    metrics = ["Shipments", "Total_Sales", "Total_Profit", "Avg_Lead_Time", "Avg_Profit_Margin"]
    metrics = [m for m in metrics if m in agg.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
    fig.suptitle("Factory Performance Overview", fontsize=14, fontweight="bold")
    colors = sns.color_palette(PALETTE_CAT, len(agg))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(agg["Factory"], agg[metric], color=colors)
        ax.set_title(metric.replace("_", " "), fontsize=11)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)

    save_chart("08_factory_performance.png")


# ════════════════════════════════════════════════════════════════
# STEP 9 — ROUTE PERFORMANCE ANALYSIS
# ════════════════════════════════════════════════════════════════

def step9_route_performance(df: pd.DataFrame) -> None:
    """Route-level analysis: fastest, slowest, top sales, top profit."""
    print("\n" + "=" * 60)
    print("STEP 9: Route Performance Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"

    for route_col in ["Factory -> State Route", "Factory -> Region Route"]:
        if route_col not in df.columns:
            continue
        tag = "state_route" if "State" in route_col else "region_route"

        agg = (
            df.groupby(route_col)
            .agg(
                Total_Shipments=("Order ID", "count"),
                Total_Sales=("Sales", "sum"),
                Total_Profit=("Gross Profit", "sum"),
                Avg_Lead_Time=(lt_col, "mean"),
            )
            .round(2)
            .reset_index()
        )

        save_summary(agg.sort_values("Total_Shipments", ascending=False),
                     f"route_{tag}_summary.csv")
                     

        # ── Top 10 Fastest ────────────────────────────────────
        fastest = agg.nsmallest(10, "Avg_Lead_Time")
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=fastest, y=route_col, x="Avg_Lead_Time",
                    palette="Greens_r", ax=ax, orient="h")
        ax.set_title(f"Top 10 Fastest Routes ({route_col})", fontweight="bold")
        ax.set_xlabel("Avg Lead Time (days)")
        save_chart(f"09_{tag}_fastest.png")

        # ── Top 10 Slowest ────────────────────────────────────
        slowest = agg.nlargest(10, "Avg_Lead_Time")
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=slowest, y=route_col, x="Avg_Lead_Time",
                    palette="Reds_r", ax=ax, orient="h")
        ax.set_title(f"Top 10 Slowest Routes ({route_col})", fontweight="bold")
        ax.set_xlabel("Avg Lead Time (days)")
        save_chart(f"09_{tag}_slowest.png")

        # ── Highest Sales Routes ──────────────────────────────
        top_sales = agg.nlargest(10, "Total_Sales")
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=top_sales, y=route_col, x="Total_Sales",
                    palette="Blues_r", ax=ax, orient="h")
        ax.set_title(f"Top 10 Routes by Sales ({route_col})", fontweight="bold")
        ax.set_xlabel("Total Sales ($)")
        save_chart(f"09_{tag}_top_sales.png")

        # ── Highest Profit Routes ─────────────────────────────
        top_profit = agg.nlargest(10, "Total_Profit")
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=top_profit, y=route_col, x="Total_Profit",
                    palette="Greens_d", ax=ax, orient="h")
        ax.set_title(f"Top 10 Routes by Profit ({route_col})", fontweight="bold")
        ax.set_xlabel("Total Gross Profit ($)")
        save_chart(f"09_{tag}_top_profit.png")

        print(f"\n  Top 5 Fastest {route_col} Routes:")
        print(fastest.head().to_string(index=False))
        print(f"\n  Top 5 Slowest {route_col} Routes:")
        print(slowest.head().to_string(index=False))

    # save_summary(agg.sort_values("Total_Sales", ascending=False), "route_summary.csv")


# ════════════════════════════════════════════════════════════════
# STEP 10 — STATE ANALYSIS
# ════════════════════════════════════════════════════════════════

def step10_state_analysis(df: pd.DataFrame) -> None:
    """State-level shipments, sales, profit and lead time."""
    print("\n" + "=" * 60)
    print("STEP 10: State Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"
    agg = (
        df.groupby("State/Province")
        .agg(
            Shipments=("Order ID", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Avg_Lead_Time=(lt_col, "mean"),
        )
        .round(2)
        .sort_values("Shipments", ascending=False)
        .reset_index()
    )

    if "Delay Status" in df.columns:
        delay_pct = (
            df[df["Delay Status"] == "Delayed"]
            .groupby("State/Province")
            .size()
            / df.groupby("State/Province").size()
            * 100
        ).round(2).rename("Delay_Pct").reset_index()
        delay_pct.columns = ["State/Province", "Delay_Pct"]
        agg = agg.merge(delay_pct, on="State/Province", how="left")
        agg["Delay_Pct"] = agg["Delay_Pct"].fillna(0)

    save_summary(agg, "state_summary.csv")

    for metric, title, color in [
        ("Shipments", "Top 10 States by Shipments", "Blues_r"),
        ("Total_Sales", "Top 10 States by Sales", "Greens_r"),
        ("Total_Profit", "Top 10 States by Profit", "Purples_r"),
        ("Avg_Lead_Time", "Top 10 States by Avg Lead Time", "Oranges_r"),
    ]:
        if metric not in agg.columns:
            continue
        top = agg.nlargest(10, metric)
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=top, y="State/Province", x=metric, palette=color, orient="h")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(metric)
        save_chart(f"10_state_{metric.lower().replace(' ', '_')}.png")

    print("  Top 10 States by Shipments:")
    print(agg.head(10).to_string(index=False))
    print("\n  Bottom 5 States by Sales:")
    print(agg.nsmallest(5, "Total_Sales").to_string(index=False))


# ════════════════════════════════════════════════════════════════
# STEP 11 — REGION ANALYSIS
# ════════════════════════════════════════════════════════════════

def step11_region_analysis(df: pd.DataFrame) -> None:
    """Regional comparison: Atlantic, Pacific, Interior, Gulf."""
    print("\n" + "=" * 60)
    print("STEP 11: Region Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"
    agg = (
        df.groupby("Region")
        .agg(
            Shipments=("Order ID", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Avg_Lead_Time=(lt_col, "mean"),
            Avg_Margin=("Profit Margin %", "mean"),
        )
        .round(2)
        .reset_index()
    )
    save_summary(agg, "region_summary.csv")
    print(agg.to_string(index=False))

    metrics = ["Shipments", "Total_Sales", "Total_Profit", "Avg_Lead_Time"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Region Performance Comparison", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    colors = sns.color_palette(PALETTE_CAT, len(agg))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(agg["Region"], agg[metric], color=colors)
        ax.set_title(metric.replace("_", " "), fontsize=11)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{bar.get_height():,.1f}",
                ha="center", va="bottom", fontsize=9,
            )

    save_chart("11_region_comparison.png")


# ════════════════════════════════════════════════════════════════
# STEP 12 — SHIP MODE ANALYSIS
# ════════════════════════════════════════════════════════════════

def step12_ship_mode(df: pd.DataFrame) -> None:
    """Ship mode comparison across shipments, sales, profit and lead time."""
    print("\n" + "=" * 60)
    print("STEP 12: Ship Mode Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"
    agg = (
        df.groupby("Ship Mode")
        .agg(
            Shipments=("Order ID", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Avg_Lead_Time=(lt_col, "mean"),
            Avg_Margin=("Profit Margin %", "mean"),
        )
        .round(2)
        .reset_index()
    )

    if "Delay Status" in df.columns:
        delay_rate = (
            df[df["Delay Status"] == "Delayed"].groupby("Ship Mode").size()
            / df.groupby("Ship Mode").size() * 100
        ).round(2).rename("Delay_Rate_%").reset_index()
        delay_rate.columns = ["Ship Mode", "Delay_Rate_%"]
        agg = agg.merge(delay_rate, on="Ship Mode", how="left")
        # Replace NaN delay rates with 0
        agg["Delay_Rate_%"] = agg["Delay_Rate_%"].fillna(0)

    save_summary(agg, "ship_mode_summary.csv")
    print(agg.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ship Mode Comparison", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    metrics = ["Shipments", "Total_Sales", "Total_Profit", "Avg_Lead_Time"]
    colors = sns.color_palette(PALETTE_CAT, len(agg))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(agg["Ship Mode"], agg[metric], color=colors)
        ax.set_title(metric.replace("_", " "), fontsize=11)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)

    save_chart("12_ship_mode_comparison.png")

    # Stacked bar: Delay Status breakdown by Ship Mode
    if "Delay Status" in df.columns:
        pivot = (
            df.groupby(["Ship Mode", "Delay Status"])
            .size()
            .unstack(fill_value=0)
        )
        ax = pivot.plot(kind="bar", stacked=True,
                        colormap="Set1", figsize=(12, 6), edgecolor="white")
        ax.set_title("Delay Status Distribution by Ship Mode", fontweight="bold")
        ax.set_xlabel("Ship Mode")
        ax.set_ylabel("Order Count")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Delay Status")
        save_chart("12_ship_mode_delay_stacked.png")


# ════════════════════════════════════════════════════════════════
# STEP 13 — DELAY ANALYSIS
# ════════════════════════════════════════════════════════════════

def step13_delay_analysis(df: pd.DataFrame) -> None:
    """Delay status breakdown by factory, state, ship mode."""
    print("\n" + "=" * 60)
    print("STEP 13: Delay Analysis")
    print("=" * 60)

    if "Delay Status" not in df.columns:
        print("  Delay Status column not found — skipping.")
        return

    overall = df["Delay Status"].value_counts()
    overall_pct = (overall / len(df) * 100).round(2)
    print("\n  Overall Delay Status:")
    print(pd.concat([overall, overall_pct.rename("Percentage %")], axis=1).to_string())

    # Pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(overall.values, labels=overall.index, autopct="%1.1f%%",
           colors=sns.color_palette("Set2", len(overall)), startangle=140)
    ax.set_title("Overall Delay Status Distribution", fontweight="bold")
    save_chart("13_delay_overall_pie.png")

    delay_summary_rows = []

    for dim in ["Factory", "State/Province", "Ship Mode"]:
        if dim not in df.columns:
            continue
        pivot = (
            df.groupby([dim, "Delay Status"])
            .size()
            .unstack(fill_value=0)
        )
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(2)
        # Replace NaN percentages with 0
        pivot_pct = pivot_pct.fillna(0)
        pivot_pct.reset_index(inplace=True)
        delay_summary_rows.append(pivot_pct.assign(Dimension=dim))

        # Stacked bar
        ax = pivot.plot(kind="barh", stacked=True,
                        colormap="Set2", figsize=(14, 6), edgecolor="white")
        ax.set_title(f"Delay Status by {dim}", fontweight="bold")
        ax.set_xlabel("Order Count")
        ax.legend(title="Delay Status")
        save_chart(f"13_delay_by_{dim.replace('/', '_').replace(' ', '_')}.png")
        print(f"\n  Delay by {dim}:")
        print(pivot_pct.head(10).to_string(index=False))

    delay_df = pd.DataFrame({
        "Delay Status": overall.index,
        "Count": overall.values,
        "Percentage %": overall_pct.values,
    })
    save_summary(delay_df, "delay_summary.csv")


# ════════════════════════════════════════════════════════════════
# STEP 14 — ROUTE EFFICIENCY ANALYSIS
# ════════════════════════════════════════════════════════════════

def step14_route_efficiency(df: pd.DataFrame) -> None:
    """Route Efficiency Score breakdown and comparison."""
    print("\n" + "=" * 60)
    print("STEP 14: Route Efficiency Analysis")
    print("=" * 60)

    if "Route Efficiency Score" not in df.columns:
        print("  Route Efficiency Score column not found — skipping.")
        return

    overall = df["Route Efficiency Score"].value_counts()
    overall_pct = (overall / len(df) * 100).round(2)
    print("\n  Overall Route Efficiency Distribution:")
    print(pd.concat([overall, overall_pct.rename("Percentage %")], axis=1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Route Efficiency Score Analysis", fontweight="bold")
    colors = sns.color_palette("RdYlGn", len(overall))

    axes[0].bar(overall.index, overall.values, color=colors, edgecolor="white")
    axes[0].set_title("Count by Efficiency Score")
    axes[0].set_ylabel("Order Count")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].pie(overall.values, labels=overall.index, autopct="%1.1f%%",
                colors=colors, startangle=140)
    axes[1].set_title("Proportion (%)")

    save_chart("14_route_efficiency_overview.png")

    # By Factory
    if "Factory" in df.columns:
        pivot = (
            df.groupby(["Factory", "Route Efficiency Score"])
            .size()
            .unstack(fill_value=0)
        )
        ax = pivot.plot(kind="bar", stacked=True,
                        colormap="RdYlGn", figsize=(13, 6), edgecolor="white")
        ax.set_title("Route Efficiency Score by Factory", fontweight="bold")
        ax.set_xlabel("Factory")
        ax.set_ylabel("Order Count")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Efficiency Score")
        save_chart("14_route_efficiency_by_factory.png")

    # By Region
    if "Region" in df.columns:
        pivot_r = (
            df.groupby(["Region", "Route Efficiency Score"])
            .size()
            .unstack(fill_value=0)
        )
        ax = pivot_r.plot(kind="bar", stacked=True,
                          colormap="RdYlGn", figsize=(12, 5), edgecolor="white")
        ax.set_title("Route Efficiency Score by Region", fontweight="bold")
        ax.set_xlabel("Region")
        ax.set_ylabel("Order Count")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Efficiency Score")
        save_chart("14_route_efficiency_by_region.png")


# ════════════════════════════════════════════════════════════════
# STEP 15 — PRODUCT ANALYSIS
# ════════════════════════════════════════════════════════════════

def step15_product_analysis(df: pd.DataFrame) -> None:
    """Product-level sales, profit and shipment analysis."""
    print("\n" + "=" * 60)
    print("STEP 15: Product Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"
    agg = (
        df.groupby("Product Name")
        .agg(
            Shipments=("Order ID", "count"),
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Total_Units=("Units", "sum"),
            Avg_Lead_Time=(lt_col, "mean"),
            Avg_Margin=("Profit Margin %", "mean"),
        )
        .round(2)
        .sort_values("Total_Sales", ascending=False)
        .reset_index()
    )
    save_summary(agg, "product_summary.csv")
    print(agg.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Product Analysis", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    metrics = ["Total_Sales", "Total_Profit", "Shipments", "Avg_Margin"]
    palette = sns.color_palette(PALETTE_CAT, len(agg))

    for i, metric in enumerate(metrics):
        sorted_agg = agg.sort_values(metric, ascending=True)
        axes[i].barh(sorted_agg["Product Name"], sorted_agg[metric], color=palette)
        axes[i].set_title(metric.replace("_", " "), fontsize=11)
        axes[i].set_xlabel(metric)

    save_chart("15_product_analysis.png")


# ════════════════════════════════════════════════════════════════
# STEP 16 — CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════

def step16_correlation(df: pd.DataFrame) -> None:
    """Correlation matrix and heatmap for key numeric fields."""
    print("\n" + "=" * 60)
    print("STEP 16: Correlation Analysis")
    print("=" * 60)

    corr_cols = [
        "Sales", "Units", "Cost", "Gross Profit",
        "Profit Margin %", "Shipping Lead Time (Simulated)",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_df = df[corr_cols].dropna()
    corr_matrix = corr_df.corr().round(3)

    save_summary(corr_matrix.reset_index().rename(columns={"index": "Column"}),
                 "correlation_matrix.csv")
    print(corr_matrix.to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True,
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_title("Correlation Heatmap — Key Metrics", fontweight="bold")
    save_chart("16_correlation_heatmap.png")

    # Pair Plot
    fig = sns.pairplot(corr_df, diag_kind="kde",
                       plot_kws={"alpha": 0.4, "color": ACCENT_COLOR})
    fig.fig.suptitle("Pair Plot — Key Metrics", y=1.02, fontweight="bold")
    fig.savefig(os.path.join(CHARTS_DIR, "16_pair_plot.png"),
                dpi=300, bbox_inches="tight")
    plt.close("all")
    global chart_count
    chart_count += 1
    print(f"  [Chart saved] {CHARTS_DIR}/16_pair_plot.png")


# ════════════════════════════════════════════════════════════════
# STEP 17 — TREND ANALYSIS
# ════════════════════════════════════════════════════════════════

def step17_trend_analysis(df: pd.DataFrame) -> None:
    """Monthly, quarterly and yearly trend analysis."""
    print("\n" + "=" * 60)
    print("STEP 17: Trend Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"

    if "Order Date" not in df.columns:
        print("  Order Date column not found — skipping.")
        return

    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # ── Monthly ──────────────────────────────────────────────
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("YearMonth")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Shipments=("Order ID", "count"),
            Avg_Lead_Time=(lt_col, "mean"),
        )
        .round(2)
        .reset_index()
    )
    save_summary(monthly, "trend_monthly.csv")

    fig, axes = plt.subplots(4, 1, figsize=(16, 18))
    fig.suptitle("Monthly Trends", fontsize=14, fontweight="bold")
    for i, (metric, color) in enumerate([
        ("Total_Sales", ACCENT_COLOR),
        ("Total_Profit", "#27AE60"),
        ("Shipments", "#8E44AD"),
        ("Avg_Lead_Time", "#E67E22"),
    ]):
        axes[i].plot(monthly["YearMonth"], monthly[metric],
                     marker="o", color=color, linewidth=2, markersize=5)
        axes[i].set_title(metric.replace("_", " "), fontsize=11)
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].grid(alpha=0.3)

    save_chart("17_monthly_trends.png")
    print("  Monthly trend data saved.")

    # ── Quarterly ────────────────────────────────────────────
    if "Order Quarter" in df.columns and "Order Year" in df.columns:
        df["YQ"] = df["Order Year"].astype(str) + "-Q" + df["Order Quarter"].astype(str)
        quarterly = (
            df.groupby("YQ")
            .agg(
                Total_Sales=("Sales", "sum"),
                Total_Profit=("Gross Profit", "sum"),
                Shipments=("Order ID", "count"),
            )
            .round(2)
            .reset_index()
        )
        save_summary(quarterly, "trend_quarterly.csv")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Quarterly Trends", fontsize=14, fontweight="bold")
        for i, metric in enumerate(["Total_Sales", "Total_Profit", "Shipments"]):
            axes[i].bar(quarterly["YQ"], quarterly[metric],
                        color=sns.color_palette(PALETTE_CAT, len(quarterly)))
            axes[i].set_title(metric.replace("_", " "), fontsize=11)
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis="x", rotation=45)

        save_chart("17_quarterly_trends.png")

    # ── Yearly ───────────────────────────────────────────────
    if "Order Year" in df.columns:
        yearly = (
            df.groupby("Order Year")
            .agg(
                Total_Sales=("Sales", "sum"),
                Total_Profit=("Gross Profit", "sum"),
                Shipments=("Order ID", "count"),
            )
            .round(2)
            .reset_index()
        )
        save_summary(yearly, "trend_yearly.csv")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Yearly Trends", fontsize=14, fontweight="bold")
        colors = sns.color_palette(PALETTE_CAT, len(yearly))
        for i, metric in enumerate(["Total_Sales", "Total_Profit", "Shipments"]):
            axes[i].bar(yearly["Order Year"].astype(str), yearly[metric], color=colors)
            axes[i].set_title(metric.replace("_", " "), fontsize=11)
            axes[i].set_ylabel(metric)

        save_chart("17_yearly_trends.png")


# ════════════════════════════════════════════════════════════════
# STEP 18 — GEOGRAPHIC ANALYSIS
# ════════════════════════════════════════════════════════════════

def step18_geographic(df: pd.DataFrame) -> None:
    """Sales, profit and lead time by state (US records only)."""
    print("\n" + "=" * 60)
    print("STEP 18: Geographic Analysis")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"
    us_df = df[df["Is US Record"] == True].copy() if "Is US Record" in df.columns else df.copy()
    print(f"  US records used for geographic analysis: {len(us_df):,}")

    state_geo = (
        us_df.groupby("State/Province")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Gross Profit", "sum"),
            Avg_Lead_Time=(lt_col, "mean"),
            Shipments=("Order ID", "count"),
        )
        .round(2)
        .reset_index()
        .sort_values("Total_Sales", ascending=False)
    )
    save_summary(state_geo, "geographic_state_summary.csv")

    for metric, title, color in [
        ("Total_Sales",   "Sales by State (Top 20)",      "Blues_r"),
        ("Total_Profit",  "Gross Profit by State (Top 20)", "Greens_r"),
        ("Avg_Lead_Time", "Avg Lead Time by State (Top 20)", "Oranges_r"),
    ]:
        top20 = state_geo.nlargest(20, metric)
        fig, ax = plt.subplots(figsize=(13, 8))
        sns.barplot(data=top20, y="State/Province", x=metric,
                    palette=color, orient="h", ax=ax)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(metric)
        save_chart(f"18_geo_{metric.lower().replace(' ', '_')}.png")

    print("  Top 10 US States by Sales:")
    print(state_geo.head(10).to_string(index=False))

    # Factory Location Coverage (text summary)
    factory_locs = {
        "Lot's O' Nuts":     "Arizona (Southwest)",
        "Wicked Choccy's":   "Georgia (Southeast)",
        "Sugar Shack":       "Minnesota (Midwest/North)",
        "Secret Factory":    "Illinois (Midwest)",
        "The Other Factory": "Tennessee (South)",
    }
    print("\n  Factory Geographic Coverage:")
    for factory, loc in factory_locs.items():
        print(f"    {factory:25s} -> {loc}")


# ════════════════════════════════════════════════════════════════
# STEP 19 — BUSINESS BOTTLENECK DETECTION
# ════════════════════════════════════════════════════════════════

def step19_bottlenecks(df: pd.DataFrame) -> None:
    """Identify worst-performing routes, states, ship modes and delay hotspots."""
    print("\n" + "=" * 60)
    print("STEP 19: Business Bottleneck Detection")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"

    # ── Worst Performing Routes ───────────────────────────────
    if "Factory -> State Route" in df.columns:
        route_agg = (
            df.groupby("Factory -> State Route")
            .agg(
                Avg_Lead_Time=(lt_col, "mean"),
                Shipments=("Order ID", "count"),
                Total_Sales=("Sales", "sum"),
            )
            .round(2)
            .reset_index()
        )
        worst_routes = route_agg.nlargest(10, "Avg_Lead_Time")
        print("\n  Top 10 Worst Routes (Slowest Lead Time):")
        print(worst_routes.to_string(index=False))

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(data=worst_routes, y="Factory -> State Route",
                    x="Avg_Lead_Time", palette="Reds_r", orient="h", ax=ax)
        ax.set_title("Top 10 Worst Performing Routes", fontweight="bold")
        ax.set_xlabel("Avg Lead Time (days)")
        save_chart("19_bottleneck_worst_routes.png")

    # ── Worst States ──────────────────────────────────────────
    state_lt = (
        df.groupby("State/Province")[lt_col]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    state_lt.columns = ["State/Province", "Avg_Lead_Time"]
    print("\n  Top 10 Worst States by Avg Lead Time:")
    print(state_lt.to_string(index=False))

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=state_lt, y="State/Province", x="Avg_Lead_Time",
                palette="OrRd", orient="h", ax=ax)
    ax.set_title("Top 10 States with Highest Avg Lead Time", fontweight="bold")
    ax.set_xlabel("Avg Lead Time (days)")
    save_chart("19_bottleneck_worst_states.png")

    # ── Worst Ship Modes ──────────────────────────────────────
    ship_lt = (
        df.groupby("Ship Mode")[lt_col]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
    )
    ship_lt.columns = ["Ship Mode", "Avg_Lead_Time"]
    print("\n  Ship Modes by Avg Lead Time (Worst First):")
    print(ship_lt.to_string(index=False))

    # ── Highest Delay by Factory / State ─────────────────────
    if "Delay Status" in df.columns:
        for dim in ["Factory", "State/Province"]:
            delay_rate = (
                df[df["Delay Status"] == "Delayed"].groupby(dim).size()
                / df.groupby(dim).size() * 100
            ).round(2).sort_values(ascending=False).head(10).reset_index()
            delay_rate.columns = [dim, "Delay_Rate_%"]
            print(f"\n  Highest Delay Rate by {dim}:")
            print(delay_rate.to_string(index=False))

            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(data=delay_rate, y=dim, x="Delay_Rate_%",
                        palette="Reds", orient="h", ax=ax)
            ax.set_title(f"Highest Delay Rate by {dim}", fontweight="bold")
            ax.set_xlabel("Delay Rate (%)")
            save_chart(f"19_bottleneck_delay_{dim.replace('/', '_').replace(' ', '_')}.png")


# ════════════════════════════════════════════════════════════════
# STEP 20 — BUSINESS INSIGHTS
# ════════════════════════════════════════════════════════════════

def step20_business_insights(df: pd.DataFrame) -> None:
    """Auto-generate at least 20 meaningful business insights."""
    print("\n" + "=" * 60)
    print("STEP 20: Business Insights")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"

    # ── Factory insights ─────────────────────────────────────
    factory_sales = df.groupby("Factory")["Sales"].sum()
    best_factory_sales = factory_sales.idxmax()
    worst_factory_sales = factory_sales.idxmin()
    add_insight(f"Best Factory by Total Sales: {best_factory_sales} (${factory_sales.max():,.2f})")
    add_insight(f"Worst Factory by Total Sales: {worst_factory_sales} (${factory_sales.min():,.2f})")

    factory_profit = df.groupby("Factory")["Gross Profit"].sum()
    add_insight(f"Highest Profit Factory: {factory_profit.idxmax()} (${factory_profit.max():,.2f})")
    add_insight(f"Lowest Profit Factory: {factory_profit.idxmin()} (${factory_profit.min():,.2f})")

    if "Profit Margin %" in df.columns:
        factory_margin = df.groupby("Factory")["Profit Margin %"].mean()
        add_insight(f"Highest Profit Margin Factory: {factory_margin.idxmax()} ({factory_margin.max():.2f}%)")

    # ── State / Region insights ───────────────────────────────
    state_sales = df.groupby("State/Province")["Sales"].sum()
    add_insight(f"Highest Sales State: {state_sales.idxmax()} (${state_sales.max():,.2f})")
    add_insight(f"Lowest Sales State: {state_sales.idxmin()} (${state_sales.min():,.2f})")

    state_profit = df.groupby("State/Province")["Gross Profit"].sum()
    add_insight(f"Highest Profit State: {state_profit.idxmax()} (${state_profit.max():,.2f})")

    region_sales = df.groupby("Region")["Sales"].sum()
    add_insight(f"Best Region by Sales: {region_sales.idxmax()} (${region_sales.max():,.2f})")
    add_insight(f"Worst Region by Sales: {region_sales.idxmin()} (${region_sales.min():,.2f})")

    # ── Product insights ──────────────────────────────────────
    prod_sales = df.groupby("Product Name")["Sales"].sum()
    add_insight(f"Highest Selling Product: {prod_sales.idxmax()} (${prod_sales.max():,.2f})")
    add_insight(f"Lowest Selling Product: {prod_sales.idxmin()} (${prod_sales.min():,.2f})")

    prod_profit = df.groupby("Product Name")["Gross Profit"].sum()
    add_insight(f"Most Profitable Product: {prod_profit.idxmax()} (${prod_profit.max():,.2f})")

    # ── Ship Mode insights ────────────────────────────────────
    ship_lt = df.groupby("Ship Mode")[lt_col].mean()
    add_insight(f"Fastest Ship Mode (Avg Lead Time): {ship_lt.idxmin()} ({ship_lt.min():.2f} days)")
    add_insight(f"Slowest Ship Mode (Avg Lead Time): {ship_lt.idxmax()} ({ship_lt.max():.2f} days)")

    ship_vol = df["Ship Mode"].value_counts()
    add_insight(f"Most Used Ship Mode: {ship_vol.idxmax()} ({ship_vol.max():,} orders = {ship_vol.max()/len(df)*100:.1f}%)")

    # ── Route insights ────────────────────────────────────────
    if "Factory -> State Route" in df.columns:
        route_lt = df.groupby("Factory -> State Route")[lt_col].mean()
        add_insight(f"Fastest Route: {route_lt.idxmin()} ({route_lt.min():.2f} days)")
        add_insight(f"Slowest Route: {route_lt.idxmax()} ({route_lt.max():.2f} days)")

        route_vol = df["Factory -> State Route"].value_counts()
        add_insight(f"Highest Efficiency Route (Most Orders): {route_vol.idxmax()} ({route_vol.max():,} orders)")

    # ── Delay insights ────────────────────────────────────────
    if "Delay Status" in df.columns:
        delay_pct = (df["Delay Status"] == "Delayed").mean() * 100
        ontime_pct = (df["Delay Status"] == "On Time").mean() * 100
        add_insight(f"Overall On-Time Rate: {ontime_pct:.1f}%")
        add_insight(f"Overall Delayed Rate: {delay_pct:.1f}%")

        if "Factory" in df.columns:
            fac_delay = (
                df[df["Delay Status"] == "Delayed"].groupby("Factory").size()
                / df.groupby("Factory").size() * 100
            )
            add_insight(f"Highest Delay Rate Factory: {fac_delay.idxmax()} ({fac_delay.max():.1f}%)")

    # ── Financial insights ────────────────────────────────────
    total_sales = df["Sales"].sum()
    total_profit = df["Gross Profit"].sum()
    avg_margin = df["Profit Margin %"].mean() if "Profit Margin %" in df.columns else 0
    add_insight(f"Total Revenue across all orders: ${total_sales:,.2f}")
    add_insight(f"Total Gross Profit: ${total_profit:,.2f}")
    add_insight(f"Overall Average Profit Margin: {avg_margin:.2f}%")

    # ── Division insight ──────────────────────────────────────
    div_sales = df.groupby("Division")["Sales"].sum()
    add_insight(f"Dominant Division: {div_sales.idxmax()} ({div_sales.max()/total_sales*100:.1f}% of total sales)")

    # ── Volume insight ────────────────────────────────────────
    top_state_orders = df["State/Province"].value_counts().idxmax()
    top_state_cnt = df["State/Province"].value_counts().max()
    add_insight(f"Highest Order Volume State: {top_state_orders} ({top_state_cnt:,} orders)")

    # ── Canada insight ────────────────────────────────────────
    if "Is US Record" in df.columns:
        ca_pct = (~df["Is US Record"]).mean() * 100
        add_insight(f"Canadian Orders: {ca_pct:.1f}% of all records")

    print(f"\n  Total business insights generated: {len(insight_list)}")

    # Save insights to file
    insights_df = pd.DataFrame({
        "Insight #": range(1, len(insight_list) + 1),
        "Insight": insight_list,
    })
    save_summary(insights_df, "business_insights.csv")


# ════════════════════════════════════════════════════════════════
# STEP 21 — PROFESSIONAL VISUALIZATIONS (additional)
# ════════════════════════════════════════════════════════════════

def step21_professional_visualizations(df: pd.DataFrame) -> None:
    """Generate additional professional charts not covered in earlier steps."""
    print("\n" + "=" * 60)
    print("STEP 21: Additional Professional Visualizations")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"

    # ── 1. Violin Plot — Lead Time by Ship Mode ──────────────
    if "Ship Mode" in df.columns and lt_col in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=df, x="Ship Mode", y=lt_col,
                       palette=PALETTE_CAT, ax=ax, inner="quartile")
        ax.set_title("Lead Time Distribution by Ship Mode (Violin)", fontweight="bold")
        ax.set_xlabel("Ship Mode")
        ax.set_ylabel("Lead Time (days, Simulated)")
        save_chart("21_violin_lead_time_ship_mode.png")

    # ── 2. Scatter Plot — Sales vs Gross Profit ───────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter_df = df[["Sales", "Gross Profit", "Division"]].dropna()
    for div in scatter_df["Division"].unique():
        subset = scatter_df[scatter_df["Division"] == div]
        ax.scatter(subset["Sales"], subset["Gross Profit"],
                   alpha=0.4, label=div, s=20)
    ax.set_title("Sales vs Gross Profit (by Division)", fontweight="bold")
    ax.set_xlabel("Sales ($)")
    ax.set_ylabel("Gross Profit ($)")
    ax.legend(title="Division")
    save_chart("21_scatter_sales_vs_profit.png")

    # ── 3. Stacked Bar — Sales by Region + Ship Mode ──────────
    if "Region" in df.columns and "Ship Mode" in df.columns:
        pivot = df.pivot_table(
            index="Region", columns="Ship Mode",
            values="Sales", aggfunc="sum", fill_value=0,
        )
        ax = pivot.plot(kind="bar", stacked=True,
                        colormap="tab10", figsize=(12, 6), edgecolor="white")
        ax.set_title("Sales by Region and Ship Mode (Stacked)", fontweight="bold")
        ax.set_xlabel("Region")
        ax.set_ylabel("Total Sales ($)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Ship Mode", bbox_to_anchor=(1.01, 1))
        save_chart("21_stacked_sales_region_shipmode.png")

    # ── 4. Heatmap — Avg Lead Time: Factory x Region ─────────
    if "Factory" in df.columns and "Region" in df.columns and lt_col in df.columns:
        pivot_lt = df.pivot_table(
            index="Factory", columns="Region",
            values=lt_col, aggfunc="mean",
        ).round(2)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_lt, annot=True, fmt=".1f",
                    cmap="YlOrRd", linewidths=0.5, ax=ax)
        ax.set_title("Avg Lead Time: Factory × Region (days)", fontweight="bold")
        save_chart("21_heatmap_factory_region_leadtime.png")

    # ── 5. Line Chart — Sales & Profit Overlay ────────────────
    if "Order Date" in df.columns:
        tmp = df.copy()
        tmp["Order Date"] = pd.to_datetime(tmp["Order Date"], errors="coerce")
        tmp["YM"] = tmp["Order Date"].dt.to_period("M").astype(str)
        monthly = tmp.groupby("YM")[["Sales", "Gross Profit"]].sum().reset_index()

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(monthly["YM"], monthly["Sales"],
                label="Sales", color=ACCENT_COLOR, linewidth=2, marker="o", markersize=4)
        ax.plot(monthly["YM"], monthly["Gross Profit"],
                label="Gross Profit", color="#27AE60", linewidth=2,
                marker="s", markersize=4, linestyle="--")
        ax.set_title("Monthly Sales vs Gross Profit Trend", fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount ($)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.grid(alpha=0.3)
        save_chart("21_line_monthly_sales_profit.png")

    # ── 6. Count Plot — Orders by Division + Ship Mode ────────
    if "Division" in df.columns and "Ship Mode" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x="Division", hue="Ship Mode",
                      palette=PALETTE_CAT, ax=ax)
        ax.set_title("Order Count by Division and Ship Mode", fontweight="bold")
        ax.set_xlabel("Division")
        ax.set_ylabel("Order Count")
        ax.legend(title="Ship Mode")
        save_chart("21_countplot_division_shipmode.png")

    # ── 7. Box Plot — Profit Margin by Factory ────────────────
    if "Factory" in df.columns and "Profit Margin %" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x="Factory", y="Profit Margin %",
                    palette=PALETTE_CAT, ax=ax)
        ax.set_title("Profit Margin % Distribution by Factory", fontweight="bold")
        ax.set_xlabel("Factory")
        ax.set_ylabel("Profit Margin %")
        ax.tick_params(axis="x", rotation=20)
        save_chart("21_boxplot_profit_margin_factory.png")

    # ── 8. Horizontal Bar — Top 15 Routes by Shipments ────────
    if "Factory -> State Route" in df.columns:
        top_routes = (
            df["Factory -> State Route"].value_counts()
            .head(15)
            .reset_index()
        )
        top_routes.columns = ["Route", "Shipments"]
        fig, ax = plt.subplots(figsize=(13, 8))
        sns.barplot(data=top_routes, y="Route", x="Shipments",
                    palette="Blues_r", orient="h", ax=ax)
        ax.set_title("Top 15 Most Frequently Used Routes", fontweight="bold")
        ax.set_xlabel("Number of Shipments")
        save_chart("21_hbar_top15_routes.png")


# ════════════════════════════════════════════════════════════════
# STEP 22 — SAVE OUTPUTS (verified via save_chart / save_summary)
# ════════════════════════════════════════════════════════════════

def step22_save_outputs() -> None:
    print("\n" + "=" * 60)
    print("STEP 22: Save Outputs")
    print("=" * 60)
    print(f"  All charts saved in : ./{CHARTS_DIR}/")
    print(f"  All summaries saved : ./{SUMMARIES_DIR}/")
    print(f"  Total charts        : {chart_count}")
    print(f"  Total summaries     : {summary_count}")


# ════════════════════════════════════════════════════════════════
# STEP 23 — EXPORT REPORTS
# ════════════════════════════════════════════════════════════════

def step23_export_reports(df: pd.DataFrame) -> None:
    """Export all final summary reports to EDA_Summaries/ folder."""
    print("\n" + "=" * 60)
    print("STEP 23: Export Reports")
    print("=" * 60)

    lt_col = "Shipping Lead Time (Simulated)"

    def grp(col, agg_dict):
        return df.groupby(col).agg(**agg_dict).round(2).reset_index()

    base = dict(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Gross Profit", "sum"),
        Shipments=("Order ID", "count"),
        Avg_Lead_Time=(lt_col, "mean"),
        Avg_Margin=("Profit Margin %", "mean"),
    )

    exports = {
        "final_sales_stats.csv":    df[["Sales"]].describe().round(2),
        "final_profit_stats.csv":   df[["Gross Profit", "Profit Margin %"]].describe().round(2),
        "final_factory_report.csv":  grp("Factory", base).sort_values("Total_Sales", ascending=False),
        "final_route_report.csv":    grp("Factory -> State Route", base).sort_values("Total_Sales", ascending=False),
        "final_state_report.csv":    grp("State/Province", base).sort_values("Total_Sales", ascending=False),
        "final_region_report.csv":   grp("Region", base).sort_values("Total_Sales", ascending=False),
        "final_ship_mode_report.csv": grp("Ship Mode", base).sort_values("Total_Sales", ascending=False),
        "final_delay_report.csv":    (
            df["Delay Status"].value_counts()
            .rename_axis("Delay Status")
            .reset_index(name="Count")
        ),
        "final_product_report.csv":  grp("Product Name", base).sort_values("Total_Sales", ascending=False),
    }

    for filename, table in exports.items():
        path = os.path.join(SUMMARIES_DIR, filename)
        table.to_csv(path, index=False, float_format="%.2f")
        print(f"  Exported: {path}")


# ════════════════════════════════════════════════════════════════
# STEP 24 — FINAL REPORT
# ════════════════════════════════════════════════════════════════

def step24_final_report() -> None:
    print("\n" + "=" * 60)
    print("STEP 24: Final Report")
    print("=" * 60)
    print(f"  Total Charts Generated      : {chart_count}")
    print(f"  Total Summary Tables Saved  : {summary_count}")
    print(f"  Total Business Insights     : {len(insight_list)}")
    print("  Execution Completed Successfully")
    print("=" * 60)


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def main():
    df = step1_load_dataset()
    step2_data_quality(df)
    step3_univariate(df)
    step4_categorical(df)
    step5_sales(df)
    step6_profit(df)
    step7_lead_time(df)
    step8_factory_performance(df)
    step9_route_performance(df)
    step10_state_analysis(df)
    step11_region_analysis(df)
    step12_ship_mode(df)
    step13_delay_analysis(df)
    step14_route_efficiency(df)
    step15_product_analysis(df)
    step16_correlation(df)
    step17_trend_analysis(df)
    step18_geographic(df)
    step19_bottlenecks(df)
    step20_business_insights(df)
    step21_professional_visualizations(df)
    step22_save_outputs()
    step23_export_reports(df)
    step24_final_report()


if __name__ == "__main__":
    main()