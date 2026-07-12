"""
02_feature_engineering.py
--------------------------
Phase 3 - Feature Engineering
Project: Factory-to-Customer Shipping Route Efficiency Analysis
         for Nassau Candy Distributor

Input  : cleaned_nassau_candy.csv   (output of 01_data_cleaning.py)
Output : featured_nassau_candy.csv  (input for Phase 4, 5, 6, 7, 8)
         + route_summary.csv, factory_summary.csv, state_summary.csv,
           region_summary.csv, ship_mode_summary.csv
"""

import pandas as pd
import numpy as np

IN_PATH = "cleaned_nassau_candy.csv"
OUT_PATH = "featured_nassau_candy.csv"


# =================================================================
# STEP 1 - Load Cleaned Dataset
# =================================================================
def load_data(path: str) -> pd.DataFrame:
    """Load the cleaned dataset and print basic structural info."""
    try:
        data = pd.read_csv(path, parse_dates=["Order Date", "Simulated Ship Date"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"'{path}' not found. Run 01_data_cleaning.py first to generate it."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load '{path}': {e}")

    print("=" * 60)
    print("STEP 1: Dataset Loaded")
    print("=" * 60)
    print("Shape:", data.shape)
    print("\nColumn Names:\n", list(data.columns))
    print("\nData Types:\n", data.dtypes)
    print("\nFirst 5 Rows:\n", data.head())
    # 'Ship Date' is the original (now fully empty) column from Phase 1 —
    # every value is NaT because the source Ship Date was corrupted.
    # Dropping it here keeps the featured dataset free of dead columns.
    data = data.drop(columns=["Ship Date"], errors="ignore")
    return data


# =================================================================
# STEP 2 - Product to Factory Mapping
# =================================================================
PRODUCT_FACTORY_MAP = {
    # Lot's O' Nuts
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":         "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":    "Lot's O' Nuts",
    # Wicked Choccy's
    "Wonka Bar - Milk Chocolate":        "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    # Sugar Shack
    "Laffy Taffy":                       "Sugar Shack",
    "SweeTARTS":                         "Sugar Shack",
    "Nerds":                             "Sugar Shack",
    "Fun Dip":                           "Sugar Shack",
    "Fizzy Lifting Drinks":              "Sugar Shack",
    # Secret Factory
    "Everlasting Gobstopper":            "Secret Factory",
    "Lickable Wallpaper":                "Secret Factory",
    "Wonka Gum":                         "Secret Factory",
    # The Other Factory
    "Hair Toffee":                       "The Other Factory",
    "Kazookles":                         "The Other Factory",
}


def map_product_to_factory(data: pd.DataFrame) -> pd.DataFrame:
    """Create the 'Factory' column from Product Name and validate mapping."""
    data["Factory"] = data["Product Name"].map(PRODUCT_FACTORY_MAP)

    missing_count = data["Factory"].isna().sum()
    unmapped_products = sorted(data.loc[data["Factory"].isna(), "Product Name"].unique())
    total_factories = data["Factory"].nunique()

    print("\n" + "=" * 60)
    print("STEP 2: Product to Factory Mapping")
    print("=" * 60)
    print("Missing Factory count:", missing_count)
    print("Unmapped Product Names:", unmapped_products if unmapped_products else "None")
    print("Total unique factories:", total_factories)

    if missing_count > 0:
        print(f"NOTE: {missing_count} rows could not be mapped to a factory "
              f"and will be excluded from route-level analysis (no origin known).")
        data = data.dropna(subset=["Factory"]).reset_index(drop=True)

    return data


# =================================================================
# STEP 3 - Factory Coordinates
# =================================================================
FACTORY_COORDS = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176, -81.088371),
    "Sugar Shack":       (48.119141, -96.181150),
    "Secret Factory":    (41.446333, -90.565487),
    "The Other Factory": (35.117500, -89.971107),
}


def add_factory_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Attach Factory Latitude / Longitude and validate completeness."""
    data["Factory Latitude"] = data["Factory"].map(lambda f: FACTORY_COORDS.get(f, (np.nan, np.nan))[0])
    data["Factory Longitude"] = data["Factory"].map(lambda f: FACTORY_COORDS.get(f, (np.nan, np.nan))[1])

    missing_coords = data["Factory Latitude"].isna().sum()

    print("\n" + "=" * 60)
    print("STEP 3: Factory Coordinates")
    print("=" * 60)
    print("Rows missing Factory coordinates:", missing_coords)
    if missing_coords == 0:
        print("Validation passed: every Factory has coordinates.")

    return data


# =================================================================
# STEP 4 - Route Engineering
# =================================================================
def add_route_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create route, customer-location and factory-location columns."""
    data["Factory -> State Route"] = data["Factory"] + " -> " + data["State/Province"]
    data["Factory -> Region Route"] = data["Factory"] + " -> " + data["Region"]
    data["Customer Location"] = data["City"] + ", " + data["State/Province"]
    data["Factory Location"] = data["Factory"]

    print("\n" + "=" * 60)
    print("STEP 4: Route Engineering")
    print("=" * 60)
    print("Unique Factory->State routes:", data["Factory -> State Route"].nunique())
    print("Unique Factory->Region routes:", data["Factory -> Region Route"].nunique())

    return data


# =================================================================
# STEP 5 - Shipping Features (Delay Status)
# =================================================================
def classify_delay_status(lead_time) -> str:
    """Classify a shipment as On Time / Moderate Delay / Delayed."""
    try:
        if pd.isna(lead_time):
            return np.nan
        if lead_time <= 3:
            return "On Time"
        elif lead_time <= 5:
            return "Moderate Delay"
        else:
            return "Delayed"
    except TypeError:
        return np.nan


def add_delay_status(data: pd.DataFrame) -> pd.DataFrame:
    """Add Delay Status using Shipping Lead Time (Simulated)."""
    data["Delay Status"] = data["Shipping Lead Time (Simulated)"].apply(classify_delay_status)

    print("\n" + "=" * 60)
    print("STEP 5: Shipping Features - Delay Status")
    print("=" * 60)
    print(data["Delay Status"].value_counts())

    return data


# =================================================================
# STEP 6 - Route Efficiency Score
# =================================================================
def classify_efficiency_score(lead_time) -> str:
    """Classify route efficiency based on shipping lead time."""
    try:
        if pd.isna(lead_time):
            return np.nan
        if lead_time <= 2:
            return "Excellent"
        elif lead_time <= 4:
            return "Good"
        elif lead_time <= 6:
            return "Average"
        else:
            return "Poor"
    except TypeError:
        return np.nan


def add_route_efficiency_score(data: pd.DataFrame) -> pd.DataFrame:
    """Add Route Efficiency Score column."""
    data["Route Efficiency Score"] = data["Shipping Lead Time (Simulated)"].apply(classify_efficiency_score)

    print("\n" + "=" * 60)
    print("STEP 6: Route Efficiency Score")
    print("=" * 60)
    print(data["Route Efficiency Score"].value_counts())

    return data


# =================================================================
# STEP 7 - Time Features
# =================================================================
def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features from Order Date and Simulated Ship Date."""
    # Order Date features
    data["Order Year"] = data["Order Date"].dt.year
    data["Order Quarter"] = data["Order Date"].dt.quarter
    data["Order Month Number"] = data["Order Date"].dt.month
    data["Order Month Name"] = data["Order Date"].dt.month_name()
    data["Order Week Number"] = data["Order Date"].dt.isocalendar().week.astype(int)
    data["Order Day"] = data["Order Date"].dt.day
    data["Order Day Name"] = data["Order Date"].dt.day_name()

    # Simulated Ship Date features
    data["Ship Year"] = data["Simulated Ship Date"].dt.year
    data["Ship Quarter"] = data["Simulated Ship Date"].dt.quarter
    data["Ship Month Number"] = data["Simulated Ship Date"].dt.month
    data["Ship Month Name"] = data["Simulated Ship Date"].dt.month_name()
    data["Ship Week Number"] = data["Simulated Ship Date"].dt.isocalendar().week.astype(int)
    data["Ship Day"] = data["Simulated Ship Date"].dt.day
    data["Ship Day Name"] = data["Simulated Ship Date"].dt.day_name()

    print("\n" + "=" * 60)
    print("STEP 7: Time Features")
    print("=" * 60)
    print("Order/Ship calendar features created successfully.")

    return data


# =================================================================
# STEP 8 - Financial Features
# =================================================================
def add_financial_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create Profit Margin %, Sales Per Unit, Cost Per Unit."""
    data["Profit Margin %"] = np.where(
        data["Sales"] != 0, (data["Gross Profit"] / data["Sales"]) * 100, np.nan
    )
    data["Sales Per Unit"] = np.where(
        data["Units"] != 0, data["Sales"] / data["Units"], np.nan
    )
    data["Cost Per Unit"] = np.where(
        data["Units"] != 0, data["Cost"] / data["Units"], np.nan
    )

    print("\n" + "=" * 60)
    print("STEP 8: Financial Features")
    print("=" * 60)
    print(data[["Profit Margin %", "Sales Per Unit", "Cost Per Unit"]].describe().round(2))

    return data


# =================================================================
# STEP 9 - Route Aggregation Features
# =================================================================
def build_route_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Build a route-level (Factory -> State) summary table."""
    route_summary = data.groupby("Factory -> State Route").agg(
        **{
            "Total Shipments": ("Order ID", "nunique"),
            "Total Sales": ("Sales", "sum"),
            "Total Units": ("Units", "sum"),
            "Total Gross Profit": ("Gross Profit", "sum"),
            "Total Cost": ("Cost", "sum"),
            "Average Shipping Lead Time (Simulated)": ("Shipping Lead Time (Simulated)", "mean"),
            "Lead Time Variability (Std Dev)": ("Shipping Lead Time (Simulated)", "std"),   # ← NEW LINE
            "Average Profit Margin": ("Profit Margin %", "mean"),
            "Average Sales": ("Sales", "mean"),
            "Average Units": ("Units", "mean"),
        }
    ).round(2).reset_index().sort_values("Total Shipments", ascending=False)

    print("\n" + "=" * 60)
    print("STEP 9: Route Aggregation Summary (Top 5)")
    print("=" * 60)
    print(route_summary.head())

    return route_summary


# =================================================================
# STEP 10 - Factory Summary
# =================================================================
def build_factory_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Build a factory-wise summary table."""
    factory_summary = data.groupby("Factory").agg(
        **{
            "Shipments": ("Order ID", "nunique"),
            "Sales": ("Sales", "sum"),
            "Units": ("Units", "sum"),
            "Gross Profit": ("Gross Profit", "sum"),
            "Average Lead Time": ("Shipping Lead Time (Simulated)", "mean"),
            "Average Profit Margin": ("Profit Margin %", "mean"),
        }
    ).round(2).reset_index().sort_values("Shipments", ascending=False)

    print("\n" + "=" * 60)
    print("STEP 10: Factory Summary")
    print("=" * 60)
    print(factory_summary)

    return factory_summary


# =================================================================
# STEP 11 - State Summary
# =================================================================
def build_state_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Build a state-wise summary table."""
    state_summary = data.groupby("State/Province").agg(
        **{
            "Shipments": ("Order ID", "nunique"),
            "Sales": ("Sales", "sum"),
            "Units": ("Units", "sum"),
            "Gross Profit": ("Gross Profit", "sum"),
            "Average Lead Time": ("Shipping Lead Time (Simulated)", "mean"),
            "Average Profit Margin": ("Profit Margin %", "mean"),
        }
    ).round(2).reset_index().sort_values("Shipments", ascending=False)

    print("\n" + "=" * 60)
    print("STEP 11: State Summary (Top 5)")
    print("=" * 60)
    print(state_summary.head())

    return state_summary


# =================================================================
# STEP 12 - Region Summary
# =================================================================
def build_region_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Build a region-wise summary table."""
    region_summary = data.groupby("Region").agg(
        **{
            "Shipments": ("Order ID", "nunique"),
            "Sales": ("Sales", "sum"),
            "Units": ("Units", "sum"),
            "Gross Profit": ("Gross Profit", "sum"),
            "Average Lead Time": ("Shipping Lead Time (Simulated)", "mean"),
            "Average Profit Margin": ("Profit Margin %", "mean"),
        }
    ).round(2).reset_index().sort_values("Shipments", ascending=False)

    print("\n" + "=" * 60)
    print("STEP 12: Region Summary")
    print("=" * 60)
    print(region_summary)

    return region_summary


# =================================================================
# STEP 13 - Ship Mode Summary
# =================================================================
def build_ship_mode_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Build a ship-mode-wise summary table."""
    ship_mode_summary = data.groupby("Ship Mode").agg(
        **{
            "Shipments": ("Order ID", "nunique"),
            "Average Lead Time": ("Shipping Lead Time (Simulated)", "mean"),
            "Sales": ("Sales", "sum"),
            "Profit": ("Gross Profit", "sum"),
        }
    ).round(2).reset_index().sort_values("Shipments", ascending=False)

    print("\n" + "=" * 60)
    print("STEP 13: Ship Mode Summary")
    print("=" * 60)
    print(ship_mode_summary)

    return ship_mode_summary


# =================================================================
# STEP 14 - Data Validation (existing, dataset-level)
# =================================================================
def validate_data(data: pd.DataFrame) -> None:
    """Print a final validation summary of the featured dataset."""
    print("\n" + "=" * 60)
    print("STEP 14: Data Validation")
    print("=" * 60)
    print("Dataset Shape:", data.shape)
    print("Number of Products:", data["Product Name"].nunique())
    print("Number of Factories:", data["Factory"].nunique())
    print("Number of States:", data["State/Province"].nunique())
    print("Number of Regions:", data["Region"].nunique())
    print("Number of Routes (Factory->State):", data["Factory -> State Route"].nunique())
    print("Number of Ship Modes:", data["Ship Mode"].nunique())
    print("\nDelay Status Distribution:\n", data["Delay Status"].value_counts())
    print("\nRoute Efficiency Distribution:\n", data["Route Efficiency Score"].value_counts())
    print("\nMissing Factory Count:", data["Factory"].isna().sum())
    print("Missing Coordinates Count:", data["Factory Latitude"].isna().sum())


# =================================================================
# STEP 15 - Save Output (existing)
# =================================================================
def save_output(data: pd.DataFrame, path: str) -> None:
    """Save the final featured dataset to CSV."""
    try:
        data.to_csv(path, index=False)
        print("\n" + "=" * 60)
        print("STEP 15: Save Output")
        print("=" * 60)
        print(f"Featured dataset saved successfully as: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save '{path}': {e}")


# =================================================================
# NEW STEP 16 - Factory ID
# =================================================================
FACTORY_ID_MAP = {
    "Lot's O' Nuts":     "F001",
    "Wicked Choccy's":   "F002",
    "Sugar Shack":       "F003",
    "Secret Factory":    "F004",
    "The Other Factory": "F005",
}


def add_factory_id(data: pd.DataFrame) -> pd.DataFrame:
    """Create a standardized Factory ID column from the Factory name."""
    data["Factory ID"] = data["Factory"].map(FACTORY_ID_MAP)

    print("\n" + "=" * 60)
    print("STEP 16: Factory ID")
    print("=" * 60)
    print("Missing Factory ID count:", data["Factory ID"].isna().sum())
    print("Unique Factory IDs:", sorted(data["Factory ID"].dropna().unique()))

    return data


# =================================================================
# NEW STEP 17 - Route ID
# =================================================================
def add_route_id(data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a unique, stable Route ID (R0001, R0002, ...) to every unique
    Factory -> State Route. Routes are sorted alphabetically first so the
    same route always receives the same ID across repeated runs.
    """
    unique_routes = sorted(data["Factory -> State Route"].dropna().unique())
    route_id_map = {route: f"R{str(i + 1).zfill(4)}" for i, route in enumerate(unique_routes)}

    data["Route ID"] = data["Factory -> State Route"].map(route_id_map)

    print("\n" + "=" * 60)
    print("STEP 17: Route ID")
    print("=" * 60)
    print("Total unique Route IDs assigned:", len(unique_routes))
    print("Sample mapping:", dict(list(route_id_map.items())[:5]))

    return data


# =================================================================
# NEW STEP 18 - Summary CSV Export
# =================================================================
def export_summary_csvs(route_summary: pd.DataFrame,
                         factory_summary: pd.DataFrame,
                         state_summary: pd.DataFrame,
                         region_summary: pd.DataFrame,
                         ship_mode_summary: pd.DataFrame) -> list:
    """Export each summary table as its own formatted CSV file."""
    exports = {
        "route_summary.csv": route_summary,
        "factory_summary.csv": factory_summary,
        "state_summary.csv": state_summary,
        "region_summary.csv": region_summary,
        "ship_mode_summary.csv": ship_mode_summary,
    }

    exported_files = []
    print("\n" + "=" * 60)
    print("STEP 18: Summary CSV Export")
    print("=" * 60)

    for filename, table in exports.items():
        try:
            table.to_csv(filename, index=False, float_format="%.2f")
            exported_files.append(filename)
            print(f"Exported: {filename}  (shape: {table.shape})")
        except Exception as e:
            print(f"WARNING: Failed to export {filename}: {e}")

    return exported_files


# =================================================================
# NEW STEP 19 - Final Column Ordering
# =================================================================
FINAL_COLUMN_ORDER = [
    "Row ID", "Order ID", "Order Date", "Simulated Ship Date", "Customer ID",
    "Country/Region", "City", "State/Province", "Region", "Postal Code",
    "Division", "Product ID", "Product Name",
    "Factory ID", "Factory", "Factory Latitude", "Factory Longitude", "Factory Location",
    "Factory -> State Route", "Factory -> Region Route", "Route ID",
    "Customer Location", "Ship Mode",
    "Shipping Lead Time", "Shipping Lead Time (Simulated)",
    "Delay Status", "Route Efficiency Score",
    "Order Year", "Order Quarter", "Order Month Number", "Order Month Name",
    "Order Week Number", "Order Day", "Order Day Name",
    "Ship Year", "Ship Quarter", "Ship Month Number", "Ship Month Name",
    "Ship Week Number", "Ship Day", "Ship Day Name",
    "Sales", "Units", "Cost", "Gross Profit",
    "Sales Per Unit", "Cost Per Unit", "Profit Margin %",
    "Ship Date Valid", "Is US Record",
]


def reorder_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Arrange columns in a clean, business-friendly order. Any column from
    FINAL_COLUMN_ORDER that exists in the dataframe is placed first, in
    that order; any extra/unexpected columns are appended at the end so
    no data is ever silently dropped.
    """
    existing_ordered = [col for col in FINAL_COLUMN_ORDER if col in data.columns]
    missing_from_order = [col for col in FINAL_COLUMN_ORDER if col not in data.columns]
    leftover_cols = [col for col in data.columns if col not in FINAL_COLUMN_ORDER]

    print("\n" + "=" * 60)
    print("STEP 19: Final Column Ordering")
    print("=" * 60)
    if missing_from_order:
        print("NOTE: expected columns not found in data (skipped):", missing_from_order)
    if leftover_cols:
        print("NOTE: extra columns appended at the end:", leftover_cols)

    data = data[existing_ordered + leftover_cols]
    print("Final column order applied. Total columns:", len(data.columns))

    return data


# =================================================================
# NEW STEP 20 - Output Validation
# =================================================================
def validate_output(data: pd.DataFrame, exported_files: list) -> None:
    """Print final validation details after all enhancements are applied."""
    print("\n" + "=" * 60)
    print("STEP 20: Output Validation")
    print("=" * 60)
    print("Number of unique Factory IDs:", data["Factory ID"].nunique())
    print("Number of unique Route IDs:", data["Route ID"].nunique())
    print("Names of all exported CSV files:", exported_files + [OUT_PATH])
    print("Final dataset shape:", data.shape)


# =================================================================
# MAIN PIPELINE
# =================================================================
def main():
    try:
        df = load_data(IN_PATH)
        df = map_product_to_factory(df)
        df = add_factory_coordinates(df)
        df = add_route_features(df)
        df = add_delay_status(df)
        df = add_route_efficiency_score(df)
        df = add_time_features(df)
        df = add_financial_features(df)

        # Aggregation tables (built from the pre-reorder dataframe;
        # column order has no effect on groupby results)
        route_summary = build_route_summary(df)
        factory_summary = build_factory_summary(df)
        state_summary = build_state_summary(df)
        region_summary = build_region_summary(df)
        ship_mode_summary = build_ship_mode_summary(df)

        validate_data(df)

        # ---- New enhancements ----
        df = add_factory_id(df)
        df = add_route_id(df)
        exported_files = export_summary_csvs(
            route_summary, factory_summary, state_summary,
            region_summary, ship_mode_summary
        )
        df = reorder_columns(df)
        validate_output(df, exported_files)

        save_output(df, OUT_PATH)

        return df, route_summary, factory_summary, state_summary, region_summary, ship_mode_summary

    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        raise


if __name__ == "__main__":
    main()