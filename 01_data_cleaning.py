"""
01_data_cleaning.py
--------------------
Phase 1 (Understand Dataset) + Phase 2 (Data Cleaning)
Project: Factory-to-Customer Shipping Route Efficiency Analysis
         for Nassau Candy Distributor

Run this file first. It reads the raw CSV, cleans it, and saves
'cleaned_nassau_candy.csv' which every later phase will use.
"""

import pandas as pd
import numpy as np

# -------------------------------------------------------------
# STEP 1: Load the raw dataset
# -------------------------------------------------------------
RAW_PATH = "Nassau_Candy_Distributor.csv"
df = pd.read_csv(RAW_PATH)

print("STEP 1: Raw data loaded")
print("Shape:", df.shape)
print(df.dtypes)

# -------------------------------------------------------------
# STEP 2: Basic data quality checks
# -------------------------------------------------------------
print("\nSTEP 2: Quality checks")
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Row ID is excluded from the duplicate check because it is a
# sequential unique key (1, 2, 3...) — if left in, no two rows
# could ever be flagged as duplicates, even if every other column
# was identical. Excluding it lets us catch TRUE duplicate entries.
before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "Row ID"])
print(f"Dropped {before - len(df)} true duplicate rows")

# -------------------------------------------------------------
# STEP 3: Convert Order Date and Ship Date to real datetime values
# -------------------------------------------------------------
# Dates in this file are in DD-MM-YYYY text format, e.g. "03-01-2024"
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")

# Any row where the date string didn't parse becomes NaT (Not a Time).
# Treat those as invalid records and remove them.
before = len(df)
df = df.dropna(subset=["Order Date", "Ship Date"])
print(f"\nSTEP 3: Dropped {before - len(df)} rows with unparseable dates")

# -------------------------------------------------------------
# STEP 4: Handle the broken/synthetic Ship Date problem
# -------------------------------------------------------------
# ASSUMPTION (documented): A real-world shipping lead time for a candy
# distributor is realistically between 0 and 30 days. Anything outside
# that range is treated as an invalid/synthetic Ship Date and is EXCLUDED
# from lead-time calculations, but the order row itself is kept for
# sales/profit/product analysis (only the lead-time-dependent fields
# become null for those rows).
MIN_VALID_LEAD_DAYS = 0
MAX_VALID_LEAD_DAYS = 30

df["Shipping Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

invalid_lead_mask = (
    (df["Shipping Lead Time"] < MIN_VALID_LEAD_DAYS)
    | (df["Shipping Lead Time"] > MAX_VALID_LEAD_DAYS)
)

n_invalid = invalid_lead_mask.sum()
n_total = len(df)
print(f"\nSTEP 4: Ship Date validity check")
print(f"  Valid lead times (0-{MAX_VALID_LEAD_DAYS} days): {n_total - n_invalid}")
print(f"  Invalid/synthetic lead times: {n_invalid} ({n_invalid/n_total:.1%} of rows)")

# Add a flag column so the issue is traceable in every later phase
df["Ship Date Valid"] = ~invalid_lead_mask

# RESULT: every single row failed this check (100% invalid). This means
# the Ship Date column is not a real date field at all, just broken data
# (root cause: it was generated relative to today's system date, not the
# actual historical shipment date). This is reported as the project's
# HEADLINE DATA QUALITY FINDING in the research paper.
#
# Decision taken for this project: since real lead times cannot be
# recovered, a SIMULATED Shipping Lead Time is generated below, for
# demonstration purposes only, so that downstream phases (route
# efficiency, ship mode comparison, dashboards) still have a lead-time
# field to work with. This simulated field is clearly labelled and
# is NEVER presented as real/observed data.
df.loc[invalid_lead_mask, "Shipping Lead Time"] = np.nan
df.loc[invalid_lead_mask, "Ship Date"] = pd.NaT

# -------------------------------------------------------------
# STEP 4b: Simulate a realistic Shipping Lead Time (CLEARLY LABELLED)
# -------------------------------------------------------------
# Typical real-world lead time ranges per ship mode (days), used only
# because the real Ship Date column is unusable:
#   Same Day        -> 0 to 1 day
#   First Class      -> 1 to 3 days
#   Second Class      -> 3 to 5 days
#   Standard Class      -> 4 to 7 days
SIMULATED_LEAD_RANGES = {
    "Same Day": (0, 1),
    "First Class": (1, 3),
    "Second Class": (3, 5),
    "Standard Class": (4, 7),
}

rng = np.random.default_rng(seed=42)  # fixed seed -> reproducible results

def simulate_lead_time(ship_mode: str) -> int:
    low, high = SIMULATED_LEAD_RANGES.get(ship_mode, (2, 6))
    return int(rng.integers(low, high + 1))

df["Shipping Lead Time (Simulated)"] = df["Ship Mode"].apply(simulate_lead_time)

df["Simulated Ship Date"] = (
    df["Order Date"] +
    pd.to_timedelta(df["Shipping Lead Time (Simulated)"], unit="D")
)

print("\nSTEP 4b: Simulated Shipping Lead Time generated")
print(df.groupby("Ship Mode")["Shipping Lead Time (Simulated)"].mean().round(2))
print("NOTE: 'Shipping Lead Time' (real) is NaN for all rows.")
print("      'Shipping Lead Time (Simulated)' is the field used for")
print("      all later Route Efficiency / Delay analysis phases.")

# -------------------------------------------------------------
# STEP 5: Fix incorrect data types
# -------------------------------------------------------------
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Units"] = pd.to_numeric(df["Units"], errors="coerce").astype("Int64")
df["Gross Profit"] = pd.to_numeric(df["Gross Profit"], errors="coerce")
df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")

# Drop rows where core financial fields failed to convert
before = len(df)
df = df.dropna(subset=["Sales", "Units", "Gross Profit", "Cost"])
print(f"\nSTEP 5: Dropped {before - len(df)} rows with bad numeric values")

# -------------------------------------------------------------
# STEP 6: Standardize text fields (state, region, product names)
# -------------------------------------------------------------
text_cols = ["Country/Region", "City", "State/Province", "Division",
             "Region", "Product Name", "Ship Mode"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()

# Separate true US states from Canadian provinces so US-only
# geographic analysis (maps, US heatmaps) is not skewed
CANADIAN_PROVINCES = {
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan"
}
df["Is US Record"] = ~df["State/Province"].isin(CANADIAN_PROVINCES)

print("\nSTEP 6: Standardized text fields")
print("US records:", df["Is US Record"].sum(), " | Canadian records:", (~df["Is US Record"]).sum())

# Fix missing leading zeros on US postal codes (e.g. Massachusetts 02151
# gets read as the number 2151, dropping the leading zero)
df.loc[df["Is US Record"], "Postal Code"] = (
    df.loc[df["Is US Record"], "Postal Code"].astype(str).str.zfill(5)
)
print("Postal codes zero-padded for US records")

# -------------------------------------------------------------
# STEP 7: Remove negative or zero financial values that don't make
# business sense (negative units/sales = data entry errors)
# -------------------------------------------------------------
before = len(df)
df = df[(df["Sales"] > 0) & (df["Units"] > 0)]
print(f"\nSTEP 7: Dropped {before - len(df)} rows with non-positive Sales/Units")

# -------------------------------------------------------------
# STEP 8: Final summary and save
# -------------------------------------------------------------
print("\nSTEP 8: Final cleaned dataset")
print("Final shape:", df.shape)
print("Rows with real Shipping Lead Time:", df["Shipping Lead Time"].notna().sum(), "(expected 0 - see data quality finding)")
print("Rows with Simulated Shipping Lead Time:", df["Shipping Lead Time (Simulated)"].notna().sum())

OUT_PATH = "cleaned_nassau_candy.csv"
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved cleaned file as: {OUT_PATH}")

# -------------------------------------------------------------
# ASSUMPTIONS DOCUMENTED IN THIS SCRIPT (for the research paper /
# methodology section):
# 1. Dates were parsed as DD-MM-YYYY.
# 2. HEADLINE FINDING: Ship Date is corrupted for 100% of records
#    (root cause: Ship Date values appear to have been generated
#    relative to the current system date rather than the real
#    historical shipment date, producing lead times of 900-1600+
#    days). The 'Shipping Lead Time' (real) column is therefore NaN
#    for every row and 'Ship Date Valid' is False for every row.
# 3. DECISION: Because real lead times cannot be recovered, a
#    SIMULATED 'Shipping Lead Time (Simulated)' column was generated
#    using realistic per-Ship-Mode ranges (Same Day 0-1 day, First
#    Class 1-3 days, Second Class 3-5 days, Standard Class 4-7 days)
#    with a fixed random seed for reproducibility. This field is used
#    for demonstration purposes ONLY in Route Efficiency, Delay
#    Status, and Ship Mode comparison phases, and is always labelled
#    "(Simulated)" wherever it appears in charts, KPIs, and the
#    dashboard so it is never mistaken for observed data.
# 4. Canadian provinces are flagged separately (Is US Record = False)
#    so US geographic visuals are not distorted.
# -------------------------------------------------------------