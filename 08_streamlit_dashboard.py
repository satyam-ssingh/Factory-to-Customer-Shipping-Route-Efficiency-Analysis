"""
08_streamlit_dashboard.py
--------------------------------------------------------------------
Phase 8 - Interactive Streamlit Dashboard
Project: Factory-to-Customer Shipping Route Efficiency Analysis
         for Nassau Candy Distributor

Run with:
    streamlit run 08_streamlit_dashboard.py

Expected input files (relative to the folder this app is launched from):
    featured_nassau_candy.csv                       (from 02_feature_engineering.py)
    EDA_Summaries/dashboard_ship_mode.csv
    EDA_Summaries/dashboard_ship_mode_kpi.csv
    EDA_Summaries/dashboard_state.csv
    EDA_Summaries/dashboard_state_kpi.csv
    EDA_Summaries/dashboard_region.csv
    EDA_Summaries/dashboard_region_kpi.csv
    EDA_Summaries/dashboard_factory.csv
    EDA_Summaries/dashboard_factory_kpi.csv
    EDA_Summaries/*_business_insights.csv            (optional, Executive Summary page)
    EDA_Summaries/*_recommendations.txt               (optional, Executive Summary page)
    EDA_Summaries/*_executive_report.txt              (optional, Executive Summary page)

The dashboard degrades gracefully: if a file is missing it is skipped
with a visible warning instead of crashing the whole app. Where possible,
KPIs and charts are recomputed live from the row-level featured dataset
so that the global sidebar filters actually affect every page.
"""

import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be the first Streamlit call)
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Nassau Candy | Shipping Route Efficiency",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ════════════════════════════════════════════════════════════════

DATA_FILE = "featured_nassau_candy.csv"
SUMMARIES_DIR = "EDA_Summaries"

LEAD_TIME_COL = "Shipping Lead Time (Simulated)"
DELAY_COL = "Delay Status"
EFF_COL = "Route Efficiency Score"
MARGIN_COL = "Profit Margin %"

ACCENT = "#2C7BB6"
GREEN = "#27AE60"
RED = "#E74C3C"
ORANGE = "#E67E22"
YELLOW = "#F1C40F"
PURPLE = "#8E44AD"
GRAY = "#95A5A6"

PALETTE = px.colors.qualitative.Set2
DELAY_COLOR_MAP = {"On Time": GREEN, "Moderate Delay": YELLOW, "Delayed": RED}
EFF_ORDER = ["Excellent", "Good", "Average", "Poor"]
EFF_COLOR_MAP = {"Excellent": "#2ECC71", "Good": "#82E0AA", "Average": "#F1C40F", "Poor": "#E74C3C"}
SHIP_MODE_ORDER = ["Same Day", "First Class", "Second Class", "Standard Class"]

US_STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
    "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

PLOTLY_TEMPLATE = "plotly_white"

DASHBOARD_FILES = {
    "ship_mode": os.path.join(SUMMARIES_DIR, "dashboard_ship_mode.csv"),
    "ship_mode_kpi": os.path.join(SUMMARIES_DIR, "dashboard_ship_mode_kpi.csv"),
    "state": os.path.join(SUMMARIES_DIR, "dashboard_state.csv"),
    "state_kpi": os.path.join(SUMMARIES_DIR, "dashboard_state_kpi.csv"),
    "region": os.path.join(SUMMARIES_DIR, "dashboard_region.csv"),
    "region_kpi": os.path.join(SUMMARIES_DIR, "dashboard_region_kpi.csv"),
    "factory": os.path.join(SUMMARIES_DIR, "dashboard_factory.csv"),
    "factory_kpi": os.path.join(SUMMARIES_DIR, "dashboard_factory_kpi.csv"),
}

INSIGHT_FILES = {
    "Ship Mode": os.path.join(SUMMARIES_DIR, "ship_mode_business_insights.csv"),
    "State": os.path.join(SUMMARIES_DIR, "state_business_insights.csv"),
    "Region": os.path.join(SUMMARIES_DIR, "region_business_insights.csv"),
    "Factory": os.path.join(SUMMARIES_DIR, "factory_business_insights.csv"),
}
RECOMMENDATION_FILES = {
    "Ship Mode": os.path.join(SUMMARIES_DIR, "ship_mode_recommendations.txt"),
    "State": os.path.join(SUMMARIES_DIR, "state_recommendations.txt"),
    "Region": os.path.join(SUMMARIES_DIR, "region_recommendations.txt"),
    "Factory": os.path.join(SUMMARIES_DIR, "factory_recommendations.txt"),
}
EXEC_REPORT_FILES = {
    "Ship Mode": os.path.join(SUMMARIES_DIR, "ship_mode_executive_report.txt"),
    "State": os.path.join(SUMMARIES_DIR, "state_executive_report.txt"),
    "Region": os.path.join(SUMMARIES_DIR, "region_executive_report.txt"),
    "Factory": os.path.join(SUMMARIES_DIR, "factory_executive_report.txt"),
}

MISSING_FILES: list = []  # populated during loading, surfaced in the sidebar


# ════════════════════════════════════════════════════════════════
# CUSTOM CSS — PROFESSIONAL THEME
# ════════════════════════════════════════════════════════════════

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main { background-color: #F7F9FB; }
        MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .app-header {
            background: linear-gradient(90deg, #1B4965 0%, #2C7BB6 55%, #5FA8D3 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        }
        .app-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
        .app-header p { margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

        div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E7ECF1;
        border-radius: 12px;
        padding: 0.9rem 1rem 0.6rem 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] * {
        color: #1B4965 !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-weight: 600 !important;
        color: #4A5568 !important;
    }
    div[data-testid="stMetricValue"] div {
        font-weight: 700 !important;
        color: #1B4965 !important;
    }

        section[data-testid="stSidebar"] {
            background-color: #10243E;
        }
        section[data-testid="stSidebar"] * { color: #EAF2F8 !important; }
        section[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1B4965;
            border-left: 5px solid #2C7BB6;
            padding-left: 0.6rem;
            margin: 1.0rem 0 0.6rem 0;
        }
        .insight-card {
            background-color: #FFFFFF;
            border-left: 4px solid #2C7BB6;
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            margin-bottom: 0.5rem;
            font-size: 0.92rem;
            color: #2D3748;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }
        .footer-note {
            text-align: center;
            color: #8896A6;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 0.8rem;
            border-top: 1px solid #E2E8F0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
# DATA LOADING (CACHED, ERROR-SAFE)
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading main dataset...")
def load_main_data(path: str) -> pd.DataFrame:
    """Load the row-level featured dataset. Returns empty df on failure."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        date_cols = [c for c in ["Order Date", "Simulated Ship Date"] if True]
        df = pd.read_csv(path, low_memory=False)
        for col in ["Order Date", "Simulated Ship Date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read '{path}': {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_csv_safe(path: str) -> pd.DataFrame:
    """Load an auxiliary CSV. Returns empty df (and records missing file) on failure."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        for col in df.columns:
            if "Date" in col:
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore")
                except Exception:  # noqa: BLE001
                    pass
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_text_safe(path: str) -> str:
    """Load a text report file. Returns '' if missing."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return ""


def check_missing(label: str, path: str) -> None:
    if not os.path.exists(path) and path not in MISSING_FILES:
        MISSING_FILES.append(path)


# ════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ════════════════════════════════════════════════════════════════

def fmt_currency(val) -> str:
    try:
        if pd.isna(val):
            return "N/A"
        return f"${val:,.2f}"
    except Exception:  # noqa: BLE001
        return "N/A"
    
def fmt_currency_compact(val) -> str:
    """Compact currency format for narrow KPI cards, e.g. $138.3K, $2.1M."""
    try:
        if pd.isna(val):
            return "N/A"
        val = float(val)
        sign = "-" if val < 0 else ""
        val = abs(val)
        if val >= 1_000_000:
            return f"{sign}${val/1_000_000:.2f}M"
        if val >= 1_000:
            return f"{sign}${val/1_000:.1f}K"
        return f"{sign}${val:,.2f}"
    except Exception:  # noqa: BLE001
        return "N/A"


def fmt_pct(val) -> str:
    try:
        if pd.isna(val):
            return "N/A"
        return f"{val:.2f}%"
    except Exception:  # noqa: BLE001
        return "N/A"


def fmt_days(val) -> str:
    try:
        if pd.isna(val):
            return "N/A"
        return f"{val:.2f} days"
    except Exception:  # noqa: BLE001
        return "N/A"
    
def fmt_days_compact(val) -> str:
    """Compact day format for narrow KPI cards, e.g. 4.40d instead of 4.40 days."""
    try:
        if pd.isna(val):
            return "N/A"
        return f"{val:.2f}Day"
    except Exception:  # noqa: BLE001
        return "N/A"


def fmt_int(val) -> str:
    try:
        if pd.isna(val):
            return "N/A"
        return f"{int(val):,}"
    except Exception:  # noqa: BLE001
        return "N/A"


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "⬇️ Download CSV", key: str = None) -> None:
    if df is None or df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv_bytes, file_name=filename, mime="text/csv", key=key)


def order_categories(values, order_list) -> list:
    values = list(values)
    known = [v for v in order_list if v in values]
    extra = [v for v in values if v not in order_list]
    return known + extra


# ════════════════════════════════════════════════════════════════
# GROUP SUMMARY / RANKING ENGINE (computed live from filtered data)
# ════════════════════════════════════════════════════════════════

def build_group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Build a per-group KPI + composite ranking table on the fly from the
    (already filtered) row-level dataset. Mirrors the logic used in the
    upstream 04/05/06/07 analysis scripts so the dashboard stays
    consistent even after filters are applied.
    """
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=[group_col])
    if work.empty:
        return pd.DataFrame()

    agg_dict = {
        "Total_Shipments": ("Order ID", "count"),
        "Total_Sales": ("Sales", "sum"),
        "Total_Profit": ("Gross Profit", "sum"),
    }
    if LEAD_TIME_COL in work.columns:
        agg_dict["Avg_Lead_Time"] = (LEAD_TIME_COL, "mean")
    if MARGIN_COL in work.columns:
        agg_dict["Avg_Profit_Margin"] = (MARGIN_COL, "mean")

    summary = work.groupby(group_col).agg(**agg_dict).round(2).reset_index()

    total_counts = work.groupby(group_col).size()

    if DELAY_COL in work.columns:
        delayed = work[work[DELAY_COL] == "Delayed"].groupby(group_col).size()
        summary["Delay_Rate_%"] = summary[group_col].map((delayed / total_counts * 100).round(2)).fillna(0.0)
        ontime = work[work[DELAY_COL] == "On Time"].groupby(group_col).size()
        summary["On_Time_Rate_%"] = summary[group_col].map((ontime / total_counts * 100).round(2)).fillna(0.0)

    if EFF_COL in work.columns:
        good = work[work[EFF_COL].isin(["Excellent", "Good"])].groupby(group_col).size()
        summary["Route_Efficiency_%"] = summary[group_col].map((good / total_counts * 100).round(2)).fillna(0.0)

    rank_cols = []
    summary["Sales_Rank"] = summary["Total_Sales"].rank(ascending=False, method="min")
    rank_cols.append("Sales_Rank")
    summary["Profit_Rank"] = summary["Total_Profit"].rank(ascending=False, method="min")
    rank_cols.append("Profit_Rank")
    if "Avg_Lead_Time" in summary.columns:
        summary["Lead_Time_Rank"] = summary["Avg_Lead_Time"].rank(ascending=True, method="min")
        rank_cols.append("Lead_Time_Rank")
    if "Delay_Rate_%" in summary.columns:
        summary["Delay_Rank"] = summary["Delay_Rate_%"].rank(ascending=True, method="min")
        rank_cols.append("Delay_Rank")
    if "Route_Efficiency_%" in summary.columns:
        summary["Efficiency_Rank"] = summary["Route_Efficiency_%"].rank(ascending=False, method="min")
        rank_cols.append("Efficiency_Rank")

    summary["Overall_Rank"] = (
        summary[rank_cols].mean(axis=1).rank(ascending=True, method="min").astype(int)
    )
    summary = summary.drop(columns=rank_cols).sort_values("Overall_Rank").reset_index(drop=True)
    return summary


def route_efficiency_distribution(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """% distribution of Route Efficiency Score per group, wide format."""
    if df.empty or EFF_COL not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    counts = df.groupby([group_col, EFF_COL]).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(2)
    for col in EFF_ORDER:
        if col not in pct.columns:
            pct[col] = 0.0
    return pct[EFF_ORDER].reset_index()


def delay_status_distribution(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Raw counts of Delay Status per group, wide format."""
    if df.empty or DELAY_COL not in df.columns or group_col not in df.columns:
        return pd.DataFrame()
    counts = df.groupby([group_col, DELAY_COL]).size().unstack(fill_value=0)
    for col in ["On Time", "Moderate Delay", "Delayed"]:
        if col not in counts.columns:
            counts[col] = 0
    return counts[["On Time", "Moderate Delay", "Delayed"]].reset_index()

def render_us_choropleth(df: pd.DataFrame, metric: str = "Total_Sales") -> None:
    """Render a US choropleth map colored by the chosen metric per state."""
    if df.empty or "State/Province" not in df.columns:
        return

    us_df = df[df.get("Is US Record", True) == True].copy() if "Is US Record" in df.columns else df.copy()
    if us_df.empty:
        st.info("No US records available for the map with current filters.")
        return

    agg_dict = {"Total_Sales": ("Sales", "sum"), "Total_Profit": ("Gross Profit", "sum")}
    if LEAD_TIME_COL in us_df.columns:
        agg_dict["Avg_Lead_Time"] = (LEAD_TIME_COL, "mean")

    state_agg = us_df.groupby("State/Province").agg(**agg_dict).round(2).reset_index()

    if DELAY_COL in us_df.columns:
        total_counts = us_df.groupby("State/Province").size()
        delayed = us_df[us_df[DELAY_COL] == "Delayed"].groupby("State/Province").size()
        state_agg["Delay_Rate_%"] = state_agg["State/Province"].map(
            (delayed / total_counts * 100).round(2)
        ).fillna(0.0)

    state_agg["Code"] = state_agg["State/Province"].map(US_STATE_ABBREV)
    state_agg = state_agg.dropna(subset=["Code"])

    if state_agg.empty:
        st.info("No mappable US states in the current filter selection.")
        return

    metric_options = [c for c in ["Total_Sales", "Total_Profit", "Avg_Lead_Time", "Delay_Rate_%"] if c in state_agg.columns]
    chosen_metric = st.selectbox("Map metric", metric_options, index=0, key="us_map_metric")

    fig = px.choropleth(
        state_agg, locations="Code", locationmode="USA-states", color=chosen_metric,
        scope="usa", color_continuous_scale="Blues" if "Sales" in chosen_metric or "Profit" in chosen_metric else "Reds",
        hover_name="State/Province", title=f"US Map — {chosen_metric.replace('_', ' ')} by State",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit


def render_factory_map(df: pd.DataFrame) -> None:
    """Render factory locations on a US map, sized by shipment volume."""
    if df.empty or "Factory" not in df.columns:
        return
    if "Factory Latitude" not in df.columns or "Factory Longitude" not in df.columns:
        st.info("Factory coordinate columns not found in the dataset.")
        return

    factory_agg = df.groupby("Factory").agg(
        Latitude=("Factory Latitude", "first"),
        Longitude=("Factory Longitude", "first"),
        Total_Shipments=("Order ID", "count"),
        Total_Sales=("Sales", "sum"),
    ).round(2).reset_index()

    fig = px.scatter_geo(
        factory_agg, lat="Latitude", lon="Longitude", text="Factory",
        size="Total_Shipments", color="Factory", scope="usa",
        color_discrete_sequence=PALETTE,
        size_max=40,
        title="Factory Locations (bubble size = shipment volume)",
        template=PLOTLY_TEMPLATE,
        hover_data={"Total_Sales": ":$,.2f", "Total_Shipments": True, "Latitude": False, "Longitude": False},
    )
    fig.update_traces(
        marker=dict(
            line=dict(width=1.5, color="white"),
            sizemin=18,
            opacity=0.9,
        ),
        textposition="top center",
        textfont=dict(size=12, color="black"),
    )
    fig.update_geos(
        showland=True, landcolor="#F0F2F5",
        showcountries=True, countrycolor="#4A5568",
        showsubunits=True, subunitcolor="#2D3748", subunitwidth=1.5,
        showcoastlines=True, coastlinecolor="#4A5568",
        showlakes=True, lakecolor="#D6E4F0",
        bgcolor="white",
    )
    fig.update_layout(height=500, legend_title_text="Factory")
    st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit


# ════════════════════════════════════════════════════════════════
# FILTERS
# ════════════════════════════════════════════════════════════════

def build_sidebar_filters(df: pd.DataFrame) -> dict:
    """Render global filter widgets in the sidebar and return selected values."""
    st.sidebar.markdown("### 🔧 Global Filters")
    st.sidebar.caption("Applied across every analysis page.")

    filters = {}

    if df.empty:
        st.sidebar.info("Load data to enable filters.")
        return filters

    with st.sidebar.expander("📍 Geography", expanded=False):
        if "State/Province" in df.columns:
            filters["State/Province"] = st.multiselect(
                "State / Province", sorted(df["State/Province"].dropna().unique()), default=[]
            )
        if "Region" in df.columns:
            filters["Region"] = st.multiselect(
                "Region", sorted(df["Region"].dropna().unique()), default=[]
            )

    with st.sidebar.expander("🏭 Operations", expanded=False):
        if "Factory" in df.columns:
            filters["Factory"] = st.multiselect(
                "Factory", sorted(df["Factory"].dropna().unique()), default=[]
            )
        if "Ship Mode" in df.columns:
            filters["Ship Mode"] = st.multiselect(
                "Ship Mode", order_categories(df["Ship Mode"].dropna().unique(), SHIP_MODE_ORDER), default=[]
            )

    with st.sidebar.expander("⏱️ Performance", expanded=False):
        if DELAY_COL in df.columns:
            filters["Delay Status"] = st.multiselect(
                "Delay Status", ["On Time", "Moderate Delay", "Delayed"], default=[]
            )
        if EFF_COL in df.columns:
            filters["Route Efficiency Score"] = st.multiselect(
                "Route Efficiency Score", EFF_ORDER, default=[]
            )
        if LEAD_TIME_COL in df.columns:                              # ← NEW BLOCK START
            max_lt = int(df[LEAD_TIME_COL].max()) if df[LEAD_TIME_COL].notna().any() else 10
            filters["Lead Time Threshold"] = st.slider(
                "Max Lead Time (days)", min_value=0, max_value=max_lt,
                value=max_lt, help="Show only shipments with lead time ≤ this value"
            )                                                         # ← NEW BLOCK END

    with st.sidebar.expander("📅 Date Range", expanded=False):
        if "Order Date" in df.columns and df["Order Date"].notna().any():
            min_d = df["Order Date"].min().date()
            max_d = df["Order Date"].max().date()
            date_range = st.date_input(
                "Order Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d
            )
            filters["Order Date"] = date_range

    if st.sidebar.button("♻️ Reset Filters"):
        st.rerun()

    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the sidebar filter selections to any dataframe that shares columns."""
    if df.empty or not filters:
        return df

    filtered = df.copy()

    for col in ["State/Province", "Region", "Factory", "Ship Mode", "Delay Status", "Route Efficiency Score"]:
        values = filters.get(col)
        if values and col in filtered.columns:
            filtered = filtered[filtered[col].isin(values)]

    date_range = filters.get("Order Date")
    if date_range and "Order Date" in filtered.columns and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["Order Date"] >= pd.Timestamp(start)) & (filtered["Order Date"] <= pd.Timestamp(end))
        ]

    lt_threshold = filters.get("Lead Time Threshold")                          # ← NEW BLOCK START
    if lt_threshold is not None and LEAD_TIME_COL in filtered.columns:
        filtered = filtered[filtered[LEAD_TIME_COL] <= lt_threshold]           # ← NEW BLOCK END

    return filtered


def active_filter_summary(filters: dict) -> str:
    parts = []

    for key, val in filters.items():

        # Date ko baad me handle karenge
        if key == "Order Date":
            continue

        # Empty filter ignore
        if not val:
            continue

        # Agar list ya tuple hai
        if isinstance(val, (list, tuple)):
            parts.append(f"{key}: {', '.join(map(str, val))}")

        # Agar slider ka integer hai
        elif isinstance(val, (int, float)):
            if key == "Lead Time Threshold":
                parts.append(f"Lead Time ≤ {val} days")
            else:
                parts.append(f"{key}: {val}")

        # Agar string hai
        else:
            parts.append(f"{key}: {val}")

    # Date Range
    date_range = filters.get("Order Date")

    if (
        date_range
        and isinstance(date_range, (list, tuple))
        and len(date_range) == 2
    ):
        parts.append(f"Order Date: {date_range[0]} → {date_range[1]}")

    if parts:
        return " | ".join(parts)
    else:
        return "No filters applied — showing full dataset."


# ════════════════════════════════════════════════════════════════
# KPI CARDS
# ════════════════════════════════════════════════════════════════

def render_kpi_row(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    total_sales = df["Sales"].sum() if "Sales" in df.columns else np.nan
    total_profit = df["Gross Profit"].sum() if "Gross Profit" in df.columns else np.nan
    avg_margin = df[MARGIN_COL].mean() if MARGIN_COL in df.columns else np.nan
    avg_lead = df[LEAD_TIME_COL].mean() if LEAD_TIME_COL in df.columns else np.nan
    delay_rate = (
        (df[DELAY_COL] == "Delayed").mean() * 100 if DELAY_COL in df.columns and len(df) else np.nan
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📦 Total Orders", fmt_int(total_orders))
    c2.metric("💰 Total Sales", fmt_currency_compact(total_sales), help=fmt_currency(total_sales))
    c3.metric("📈 Total Gross Profit", fmt_currency_compact(total_profit), help=fmt_currency(total_profit))
    c4.metric("📊 Avg Profit Margin", fmt_pct(avg_margin))
    c5.metric("🚚 Avg Lead Time", fmt_days_compact(avg_lead), help=fmt_days(avg_lead))
    c6.metric("⏰ Overall Delay Rate", fmt_pct(delay_rate))


# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ════════════════════════════════════════════════════════════════

def page_overview(df: pd.DataFrame, filters: dict) -> None:
    st.markdown('<div class="section-title">🏠 Overview Dashboard</div>', unsafe_allow_html=True)
    st.caption(active_filter_summary(filters))

    render_kpi_row(df)

    if df.empty:
        return

    st.markdown('<div class="section-title">📈 Trends Over Time</div>', unsafe_allow_html=True)

    if "Order Date" in df.columns and df["Order Date"].notna().any():
        monthly = df.copy()
        monthly["YearMonth"] = monthly["Order Date"].dt.to_period("M").astype(str)

        agg_dict = {"Sales": "sum", "Gross Profit": "sum"}
        if LEAD_TIME_COL in monthly.columns:
            agg_dict[LEAD_TIME_COL] = "mean"
        trend = monthly.groupby("YearMonth").agg(agg_dict).reset_index().sort_values("YearMonth")

        if DELAY_COL in monthly.columns:
            delay_trend = (
                monthly.groupby("YearMonth")[DELAY_COL]
                .apply(lambda s: (s == "Delayed").mean() * 100)
                .reset_index(name="Delay_Rate_%")
            )
            trend = trend.merge(delay_trend, on="YearMonth", how="left")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(trend, x="YearMonth", y="Sales", markers=True, title="Sales Trend",
                          template=PLOTLY_TEMPLATE, color_discrete_sequence=[ACCENT])
            fig.update_layout(xaxis_title="Month", yaxis_title="Total Sales ($)", height=380)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

            if LEAD_TIME_COL in trend.columns:
                fig = px.line(trend, x="YearMonth", y=LEAD_TIME_COL, markers=True, title="Lead Time Trend",
                              template=PLOTLY_TEMPLATE, color_discrete_sequence=[ORANGE])
                fig.update_layout(xaxis_title="Month", yaxis_title="Avg Lead Time (days)", height=380)
                st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

        with col2:
            fig = px.line(trend, x="YearMonth", y="Gross Profit", markers=True, title="Profit Trend",
                          template=PLOTLY_TEMPLATE, color_discrete_sequence=[GREEN])
            fig.update_layout(xaxis_title="Month", yaxis_title="Total Gross Profit ($)", height=380)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

            if "Delay_Rate_%" in trend.columns:
                fig = px.line(trend, x="YearMonth", y="Delay_Rate_%", markers=True, title="Delay Trend",
                              template=PLOTLY_TEMPLATE, color_discrete_sequence=[RED])
                fig.update_layout(xaxis_title="Month", yaxis_title="Delay Rate (%)", height=380)
                st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit
    else:
        st.info("Order Date column not available — trend charts skipped.")

    st.markdown('<div class="section-title">🍩 Quick Composition</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if "Ship Mode" in df.columns:
            counts = df["Ship Mode"].value_counts().reset_index()
            counts.columns = ["Ship Mode", "Count"]
            fig = px.pie(counts, names="Ship Mode", values="Count", title="Shipments by Ship Mode",
                        color_discrete_sequence=PALETTE, hole=0.4)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit
    with col2:
        if DELAY_COL in df.columns:
            counts = df[DELAY_COL].value_counts().reset_index()
            counts.columns = ["Delay Status", "Count"]
            fig = px.pie(counts, names="Delay Status", values="Count", title="Delay Status Split",
                        color="Delay Status", color_discrete_map=DELAY_COLOR_MAP, hole=0.4)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit
    with col3:
        if "Region" in df.columns:
            counts = df["Region"].value_counts().reset_index()
            counts.columns = ["Region", "Count"]
            fig = px.pie(counts, names="Region", values="Count", title="Shipments by Region",
                        color_discrete_sequence=PALETTE, hole=0.4)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

    with st.expander("📋 View & Download Filtered Row-Level Data"):
        st.dataframe(df.head(500), use_container_width=True)
        download_csv_button(df, "nassau_candy_filtered_data.csv", key="dl_overview")


# ════════════════════════════════════════════════════════════════
# GENERIC GROUP ANALYSIS PAGE (used for Ship Mode / State / Region / Factory)
# ════════════════════════════════════════════════════════════════

def page_group_analysis(
    df: pd.DataFrame,
    group_col: str,
    page_title: str,
    icon: str,
    reference_kpi_df: pd.DataFrame,
    top_bottom: bool = False,
    default_top_n: int = 15,
) -> None:
    st.markdown(f'<div class="section-title">{icon} {page_title}</div>', unsafe_allow_html=True)

    if df.empty or group_col not in df.columns:
        st.warning(f"No data available for {page_title} with the current filters.")
        return

    summary = build_group_summary(df, group_col)
    if summary.empty:
        st.warning("Not enough data to build a summary for the current filter selection.")
        return
    
    if group_col == "State/Province":
        st.markdown('<div class="section-title">🗺️ US Choropleth Map</div>', unsafe_allow_html=True)
        render_us_choropleth(df)
    elif group_col == "Factory":
        st.markdown('<div class="section-title">📍 Factory Locations Map</div>', unsafe_allow_html=True)
        render_factory_map(df)

    n_groups = len(summary)
    top_n = st.slider(
        f"Show top / bottom N {group_col.split('/')[0]}(s)", min_value=5,
        max_value=max(5, n_groups), value=min(default_top_n, n_groups), key=f"topn_{group_col}",
    ) if top_bottom and n_groups > 5 else n_groups

    col1, col2 = st.columns(2)
    with col1:
        sales_sorted = summary.sort_values("Total_Sales", ascending=False).head(top_n)
        fig = px.bar(sales_sorted, x=group_col, y="Total_Sales", title=f"Sales by {group_col}",
                    template=PLOTLY_TEMPLATE, color="Total_Sales", color_continuous_scale="Blues")
        fig.update_layout(xaxis_tickangle=-30, height=380, showlegend=False)
        st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

        if "Avg_Lead_Time" in summary.columns:
            lt_sorted = summary.sort_values("Avg_Lead_Time", ascending=True).head(top_n)
            fig = px.bar(lt_sorted, x=group_col, y="Avg_Lead_Time", title=f"Avg Lead Time by {group_col}",
                        template=PLOTLY_TEMPLATE, color="Avg_Lead_Time", color_continuous_scale="Oranges")
            fig.update_layout(xaxis_tickangle=-30, height=380, showlegend=False)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

        if "Route_Efficiency_%" in summary.columns:
            eff_dist = route_efficiency_distribution(df, group_col)
            if not eff_dist.empty:
                eff_dist = eff_dist[eff_dist[group_col].isin(sales_sorted[group_col])]
                melted = eff_dist.melt(id_vars=group_col, var_name="Efficiency", value_name="Percent")
                fig = px.bar(melted, x=group_col, y="Percent", color="Efficiency", barmode="stack",
                            title=f"Route Efficiency Distribution by {group_col}",
                            template=PLOTLY_TEMPLATE, color_discrete_map=EFF_COLOR_MAP)
                fig.update_layout(xaxis_tickangle=-30, height=380)
                st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

    with col2:
        profit_sorted = summary.sort_values("Total_Profit", ascending=False).head(top_n)
        fig = px.bar(profit_sorted, x=group_col, y="Total_Profit", title=f"Profit by {group_col}",
                    template=PLOTLY_TEMPLATE, color="Total_Profit", color_continuous_scale="Greens")
        fig.update_layout(xaxis_tickangle=-30, height=380, showlegend=False)
        st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

        if "Delay_Rate_%" in summary.columns:
            delay_sorted = summary.sort_values("Delay_Rate_%", ascending=False).head(top_n)
            fig = px.bar(delay_sorted, x=group_col, y="Delay_Rate_%", title=f"Delay Rate (%) by {group_col}",
                        template=PLOTLY_TEMPLATE, color="Delay_Rate_%", color_continuous_scale="Reds")
            fig.update_layout(xaxis_tickangle=-30, height=380, showlegend=False)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

        shipment_sorted = summary.sort_values("Total_Shipments", ascending=False).head(top_n)
        fig = px.bar(shipment_sorted, x=group_col, y="Total_Shipments", title=f"Shipment Count by {group_col}",
                    template=PLOTLY_TEMPLATE, color="Total_Shipments", color_continuous_scale="Purp")
        fig.update_layout(xaxis_tickangle=-30, height=380, showlegend=False)
        st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

    if top_bottom:
        st.markdown('<div class="section-title">🔝 Top vs 🔻 Bottom Performers (by Sales)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            top5 = summary.sort_values("Total_Sales", ascending=False).head(5)
            fig = px.bar(top5, x="Total_Sales", y=group_col, orientation="h", title="Top 5 by Sales",
                        template=PLOTLY_TEMPLATE, color_discrete_sequence=[GREEN])
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit
        with c2:
            bottom5 = summary.sort_values("Total_Sales", ascending=True).head(5)
            fig = px.bar(bottom5, x="Total_Sales", y=group_col, orientation="h", title="Bottom 5 by Sales",
                        template=PLOTLY_TEMPLATE, color_discrete_sequence=[RED])
            fig.update_layout(yaxis={"categoryorder": "total descending"}, height=320)
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

    # ── Heatmap: group vs Delay Status ────────────────────────
    if DELAY_COL in df.columns:
        delay_dist = delay_status_distribution(df, group_col)
        if not delay_dist.empty:
            heat = delay_dist.set_index(group_col)
            heat_pct = heat.div(heat.sum(axis=1), axis=0).mul(100).round(1)
            fig = go.Figure(data=go.Heatmap(
                z=heat_pct.values, x=heat_pct.columns, y=heat_pct.index,
                colorscale="YlOrRd", text=heat_pct.values, texttemplate="%{text}",
                colorbar=dict(title="% of Shipments"),
            ))
            fig.update_layout(title=f"{group_col} × Delay Status Heatmap", template=PLOTLY_TEMPLATE,
                              height=max(320, 24 * len(heat_pct)))
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

    # ── Scatter: Sales vs Profit ───────────────────────────────
    fig = px.scatter(
        summary, x="Total_Sales", y="Total_Profit", size="Total_Shipments", color=group_col,
        title=f"Sales vs Profit by {group_col} (bubble = shipment volume)",
        template=PLOTLY_TEMPLATE, color_discrete_sequence=PALETTE,
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

    # ── Overall Ranking ──────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Overall Ranking (live, filter-aware)</div>', unsafe_allow_html=True)
    display_cols = [group_col, "Overall_Rank", "Total_Shipments", "Total_Sales", "Total_Profit"]
    for c in ["Avg_Lead_Time", "Avg_Profit_Margin", "Delay_Rate_%", "Route_Efficiency_%"]:
        if c in summary.columns:
            display_cols.append(c)
    st.dataframe(
        summary[display_cols].sort_values("Overall_Rank"),
        use_container_width=True, hide_index=True,
    )
    download_csv_button(summary, f"{group_col.replace('/', '_')}_ranking.csv", key=f"dl_rank_{group_col}")

    if not reference_kpi_df.empty:
        with st.expander("📎 Reference: Precomputed KPI Table (full, unfiltered dataset)"):
            st.dataframe(reference_kpi_df, use_container_width=True, hide_index=True)
            download_csv_button(reference_kpi_df, f"{group_col.replace('/', '_')}_kpi_reference.csv",
                               key=f"dl_ref_{group_col}")


# ════════════════════════════════════════════════════════════════
# PAGE 6 — INTERACTIVE FILTERS / DATA EXPLORER
# ════════════════════════════════════════════════════════════════

def page_filters(df_raw: pd.DataFrame, df_filtered: pd.DataFrame, filters: dict) -> None:
    st.markdown('<div class="section-title">🔍 Interactive Filters & Data Explorer</div>', unsafe_allow_html=True)
    st.write(
        "Use the **Global Filters** panel in the sidebar (State, Region, Factory, Ship Mode, "
        "Delay Status, Route Efficiency Score, Order Date range) to slice the dataset. "
        "Every page in this dashboard reacts to those filters automatically."
    )
    st.info(active_filter_summary(filters))

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows — Full Dataset", fmt_int(len(df_raw)))
    c2.metric("Rows — After Filters", fmt_int(len(df_filtered)))
    pct_kept = (len(df_filtered) / len(df_raw) * 100) if len(df_raw) else 0
    c3.metric("% of Data Retained", fmt_pct(pct_kept))

    st.markdown('<div class="section-title">📋 Filtered Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_filtered.head(1000), use_container_width=True)
    download_csv_button(df_filtered, "nassau_candy_filtered_export.csv", key="dl_filters_page")

    if not df_filtered.empty:
        st.markdown('<div class="section-title">📐 Quick Stats on Filtered Data</div>', unsafe_allow_html=True)
        numeric_cols = [c for c in ["Sales", "Units", "Cost", "Gross Profit", MARGIN_COL, LEAD_TIME_COL]
                        if c in df_filtered.columns]
        if numeric_cols:
            st.dataframe(df_filtered[numeric_cols].describe().round(2).T, use_container_width=True)

            st.markdown('<div class="section-title">🔎 Order-Level Drill-Down</div>', unsafe_allow_html=True)
    if not df_filtered.empty and "Order ID" in df_filtered.columns:
        order_ids = sorted(df_filtered["Order ID"].dropna().unique())
        selected_order = st.selectbox("Select an Order ID to inspect", order_ids, key="order_drilldown")

        order_rows = df_filtered[df_filtered["Order ID"] == selected_order]
        if not order_rows.empty:
            row = order_rows.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Factory", row.get("Factory", "N/A"))
            c2.metric("Ship Mode", row.get("Ship Mode", "N/A"))
            c3.metric("Delay Status", row.get("Delay Status", "N/A"))
            c4.metric("Lead Time", fmt_days(row.get(LEAD_TIME_COL, np.nan)))


            timeline_data = {
                "Event": ["Order Placed", "Simulated Ship Date"],
                "Date": [row.get("Order Date"), row.get("Simulated Ship Date")],
            }
            timeline_df = pd.DataFrame(timeline_data)
            fig = px.scatter(
                timeline_df, x="Date", y="Event", text="Event",
                title=f"Shipment Timeline — Order {selected_order}", template=PLOTLY_TEMPLATE,
            )
            fig.update_traces(
                marker=dict(size=16, color=ACCENT, line=dict(width=2, color="white")),
                textposition="top center",
            )
            # Connect the two points with a line to show the lead-time gap visually
            fig.add_shape(
                type="line",
                x0=timeline_df["Date"].iloc[0], x1=timeline_df["Date"].iloc[1],
                y0=0, y1=1,
                line=dict(color=GRAY, width=2, dash="dot"),
            )
            fig.update_layout(height=250, showlegend=False, yaxis_title="", xaxis_title="Date")
            st.plotly_chart(fig, width='stretch')   # Recommended for new Streamlit

            st.dataframe(order_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No orders available to drill into with the current filters.")


# ════════════════════════════════════════════════════════════════
# PAGE 7 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════

def best_row(df: pd.DataFrame, col: str, group_col: str, ascending: bool = False):
    if df.empty or col not in df.columns:
        return None
    idx = df[col].idxmin() if ascending else df[col].idxmax()
    return df.loc[idx]


def generate_text_report(df: pd.DataFrame, summaries: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("EXECUTIVE SUMMARY REPORT")
    lines.append("Factory-to-Customer Shipping Route Efficiency Analysis")
    lines.append("Nassau Candy Distributor")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)
    lines.append("")

    if not df.empty:
        lines.append("HEADLINE KPIs (current filter selection)")
        lines.append("-" * 70)
        lines.append(f"Total Orders          : {df['Order ID'].nunique() if 'Order ID' in df.columns else len(df):,}")
        lines.append(f"Total Sales           : {fmt_currency(df['Sales'].sum() if 'Sales' in df.columns else np.nan)}")
        lines.append(f"Total Gross Profit    : {fmt_currency(df['Gross Profit'].sum() if 'Gross Profit' in df.columns else np.nan)}")
        if MARGIN_COL in df.columns:
            lines.append(f"Avg Profit Margin     : {fmt_pct(df[MARGIN_COL].mean())}")
        if LEAD_TIME_COL in df.columns:
            lines.append(f"Avg Shipping Lead Time: {fmt_days(df[LEAD_TIME_COL].mean())}")
        if DELAY_COL in df.columns:
            lines.append(f"Overall Delay Rate    : {fmt_pct((df[DELAY_COL] == 'Delayed').mean() * 100)}")
        lines.append("")

    for label, (group_col, summary) in summaries.items():
        if summary.empty:
            continue
        lines.append(f"BEST {label.upper()} (by composite rank)")
        lines.append("-" * 70)
        best = summary.sort_values("Overall_Rank").iloc[0]
        lines.append(f"  {best[group_col]}  (Rank #{int(best['Overall_Rank'])})")
        lines.append(f"    Sales: {fmt_currency(best['Total_Sales'])} | Profit: {fmt_currency(best['Total_Profit'])}")
        if "Avg_Lead_Time" in summary.columns:
            lines.append(f"    Avg Lead Time: {fmt_days(best['Avg_Lead_Time'])}")
        if "Delay_Rate_%" in summary.columns:
            lines.append(f"    Delay Rate: {fmt_pct(best['Delay_Rate_%'])}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    return "\n".join(lines)


def page_executive_summary(df: pd.DataFrame, group_summaries: dict) -> None:
    st.markdown('<div class="section-title">📋 Executive Summary</div>', unsafe_allow_html=True)
    st.caption("High-level takeaways for leadership — based on the current filter selection.")

    render_kpi_row(df)

    st.markdown('<div class="section-title">🏅 Best-in-Class Highlights</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    labels = ["Ship Mode", "State", "Region", "Factory"]
    for col_widget, label in zip(cols, labels):
        group_col, summary = group_summaries.get(label, (None, pd.DataFrame()))
        with col_widget:
            if summary.empty:
                st.info(f"No {label} data")
                continue
            best = summary.sort_values("Overall_Rank").iloc[0]
            st.markdown(f"**🏆 Best {label}**")
            st.markdown(f"### {best[group_col]}")
            st.caption(
                f"Sales {fmt_currency(best['Total_Sales'])} · "
                f"Profit {fmt_currency(best['Total_Profit'])}"
            )

    if not df.empty:
        st.markdown('<div class="section-title">⭐ Overall Highest / Fastest / Lowest-Delay</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("💰 Highest Sales (row)", fmt_currency(df["Sales"].max()) if "Sales" in df.columns else "N/A")
        with c2:
            st.metric("📈 Highest Profit (row)", fmt_currency(df["Gross Profit"].max()) if "Gross Profit" in df.columns else "N/A")
        with c3:
            st.metric("🚀 Fastest Shipment", fmt_days(df[LEAD_TIME_COL].min()) if LEAD_TIME_COL in df.columns else "N/A")
        with c4:
            if DELAY_COL in df.columns:
                total = len(df)
                on_time = (df[DELAY_COL] == "On Time").sum()
                st.metric("✅ On-Time Rate", fmt_pct(on_time / total * 100 if total else np.nan))
            else:
                st.metric("✅ On-Time Rate", "N/A")

    st.markdown('<div class="section-title">💡 Business Insights (precomputed)</div>', unsafe_allow_html=True)
    tabs = st.tabs(list(INSIGHT_FILES.keys()))
    for tab, label in zip(tabs, INSIGHT_FILES.keys()):
        with tab:
            insights_df = load_csv_safe(INSIGHT_FILES[label])
            if insights_df.empty:
                check_missing(label, INSIGHT_FILES[label])
                st.info(f"No precomputed insights file found for {label}. "
                       f"Run the corresponding analysis script (04–07) to generate it.")
                continue
            insight_col = "Insight" if "Insight" in insights_df.columns else insights_df.columns[-1]
            for _, row in insights_df.iterrows():
                st.markdown(f'<div class="insight-card">🔹 {row[insight_col]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📌 Recommendations (precomputed)</div>', unsafe_allow_html=True)
    rec_tabs = st.tabs(list(RECOMMENDATION_FILES.keys()))
    for tab, label in zip(rec_tabs, RECOMMENDATION_FILES.keys()):
        with tab:
            text = load_text_safe(RECOMMENDATION_FILES[label])
            if not text:
                check_missing(label, RECOMMENDATION_FILES[label])
                st.info(f"No recommendations file found for {label}.")
                continue
            st.text(text)

    with st.expander("📜 Full Precomputed Executive Reports"):
        exec_tabs = st.tabs(list(EXEC_REPORT_FILES.keys()))
        for tab, label in zip(exec_tabs, EXEC_REPORT_FILES.keys()):
            with tab:
                text = load_text_safe(EXEC_REPORT_FILES[label])
                if not text:
                    check_missing(label, EXEC_REPORT_FILES[label])
                    st.info(f"No executive report file found for {label}.")
                    continue
                st.text(text)

    st.markdown('<div class="section-title">⬇️ Export</div>', unsafe_allow_html=True)
    report_text = generate_text_report(df, group_summaries)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📄 Download Executive Report (.txt)",
            data=report_text, file_name="Nassau_Candy_Executive_Summary.txt", mime="text/plain",
        )
    with c2:
        download_csv_button(df, "nassau_candy_current_view.csv", label="⬇️ Download Current Data (.csv)",
                           key="dl_exec_csv")


# ════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════

def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>🍬 Nassau Candy Distributor — Shipping Route Efficiency Analytics</h1>
            <p>Factory-to-Customer Shipping Route Efficiency Analysis · Interactive Logistics Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_missing_files_warning() -> None:
    if MISSING_FILES:
        with st.sidebar.expander("⚠️ Missing Optional Files", expanded=False):
            st.caption("These files were not found. Related sections were skipped gracefully.")
            for f in MISSING_FILES:
                st.write(f"• `{f}`")


def main() -> None:
    inject_custom_css()
    render_header()

    # ── Load core data ──────────────────────────────────────────
    df_raw = load_main_data(DATA_FILE)
    if df_raw.empty:
        st.error(
            f"Could not find or read '{DATA_FILE}'. Please run 01_data_cleaning.py and "
            "02_feature_engineering.py first, and launch this dashboard from the same folder."
        )
        st.stop()

    for label, path in DASHBOARD_FILES.items():
        check_missing(label, path)

    ship_mode_kpi_ref = load_csv_safe(DASHBOARD_FILES["ship_mode_kpi"])
    state_kpi_ref = load_csv_safe(DASHBOARD_FILES["state_kpi"])
    region_kpi_ref = load_csv_safe(DASHBOARD_FILES["region_kpi"])
    factory_kpi_ref = load_csv_safe(DASHBOARD_FILES["factory_kpi"])

    # ── Sidebar navigation ───────────────────────────────────────
    st.sidebar.markdown("## 🍬 Nassau Candy")
    st.sidebar.markdown("#### Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Overview Dashboard",
            "🚚 Ship Mode Analysis",
            "🗺️ State Analysis",
            "🌎 Region Analysis",
            "🏭 Factory Analysis",
            "🔍 Interactive Filters",
            "📋 Executive Summary",
        ],
        label_visibility="collapsed",
    )

    filters = build_sidebar_filters(df_raw)
    df_filtered = apply_filters(df_raw, filters)

    render_missing_files_warning()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dataset rows: {len(df_raw):,} | After filters: {len(df_filtered):,}")
    st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── Page routing (each wrapped for resilience) ────────────────
    try:
        if page == "🏠 Overview Dashboard":
            page_overview(df_filtered, filters)

        elif page == "🚚 Ship Mode Analysis":
            page_group_analysis(
                df_filtered, "Ship Mode", "Ship Mode Analysis", "🚚",
                reference_kpi_df=ship_mode_kpi_ref, top_bottom=False,
            )

        elif page == "🗺️ State Analysis":
            page_group_analysis(
                df_filtered, "State/Province", "State Analysis", "🗺️",
                reference_kpi_df=state_kpi_ref, top_bottom=True, default_top_n=15,
            )

        elif page == "🌎 Region Analysis":
            page_group_analysis(
                df_filtered, "Region", "Region Analysis", "🌎",
                reference_kpi_df=region_kpi_ref, top_bottom=False,
            )

        elif page == "🏭 Factory Analysis":
            page_group_analysis(
                df_filtered, "Factory", "Factory Analysis", "🏭",
                reference_kpi_df=factory_kpi_ref, top_bottom=False,
            )

        elif page == "🔍 Interactive Filters":
            page_filters(df_raw, df_filtered, filters)

        elif page == "📋 Executive Summary":
            group_summaries = {
                "Ship Mode": ("Ship Mode", build_group_summary(df_filtered, "Ship Mode")),
                "State": ("State/Province", build_group_summary(df_filtered, "State/Province")),
                "Region": ("Region", build_group_summary(df_filtered, "Region")),
                "Factory": ("Factory", build_group_summary(df_filtered, "Factory")),
            }
            page_executive_summary(df_filtered, group_summaries)

    except Exception as exc:  # noqa: BLE001
        st.error("Something went wrong while rendering this page.")
        st.exception(exc)

    st.markdown(
        '<div class="footer-note">'
        'Nassau Candy Distributor · Factory-to-Customer Shipping Route Efficiency Analysis '
        '· Built with Streamlit &amp; Plotly'
        '<br><strong>Developed by Satyam Kumar Singh</strong>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
