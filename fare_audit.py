
import pandas as pd
import numpy as np

from pathlib import Path

# --------------------------------------------------
# INT395: Airfare Dataset Quality Audit
# --------------------------------------------------

DATA_PATH = Path("data/Clean_Dataset.csv")

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("INT395 AIRFARE DATASET QUALITY AUDIT")
print("=" * 65)

# 1. Overall dataset quality
print("\n1. DATASET OVERVIEW")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", df.isnull().sum().sum())
print("Exact duplicate rows:", df.duplicated().sum())

# 2. Price distribution
print("\n2. PRICE DISTRIBUTION")

print(df["price"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
))

# 3. Check extreme prices using IQR
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["price"] < lower_bound)
    | (df["price"] > upper_bound)
]

print("\n3. EXTREME PRICE INVESTIGATION")

print("IQR lower bound:", round(lower_bound, 2))
print("IQR upper bound:", round(upper_bound, 2))
print("Potential statistical outliers:", len(outliers))

print("\nHighest 10 prices:")
print(
    df.nlargest(10, "price")[
        ["airline", "source_city", "destination_city",
         "class", "days_left", "price"]
    ].to_string(index=False)
)

# 4. Route-level price analysis
print("\n4. ROUTE PRICE ANALYSIS")

route_stats = (
    df.groupby(
        ["source_city", "destination_city", "class"]
    )["price"]
    .agg(["count", "min", "median", "mean", "max"])
    .reset_index()
)

print("\nRoutes with the highest median prices:")
print(
    route_stats.sort_values(
        "median", ascending=False
    ).head(10).to_string(index=False)
)

# 5. Price changes based on booking window
print("\n5. BOOKING WINDOW ANALYSIS")

booking_stats = (
    df.groupby("days_left")["price"]
    .agg(["count", "median", "mean", "min", "max"])
    .reset_index()
)

print(booking_stats.head(10).to_string(index=False))
print("\nLast 10 booking windows:")
print(booking_stats.tail(10).to_string(index=False))

# 6. Airline and class price ranges
print("\n6. AIRLINE AND CLASS PRICE RANGES")

airline_class_stats = (
    df.groupby(["airline", "class"])["price"]
    .agg(["count", "min", "median", "mean", "max"])
    .reset_index()
)

print(airline_class_stats.to_string(index=False))

# 7. Check whether duplicate flight characteristics
# have conflicting prices

comparison_columns = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left"
]

group_stats = (
    df.groupby(comparison_columns)["price"]
    .agg(["count", "min", "max", "median"])
    .reset_index()
)

conflicting_groups = group_stats[
    group_stats["min"] != group_stats["max"]
]

print("\n7. SAME-CHARACTERISTIC PRICE VARIATION")

print(
    "Groups with different prices:",
    len(conflicting_groups)
)

print(
    "Total repeated characteristic groups:",
    (group_stats["count"] > 1).sum()
)

print("\nExamples of varying prices:")
print(
    conflicting_groups.head(10).to_string(index=False)
)

# 8. Save audit results
Path("outputs").mkdir(exist_ok=True)

route_stats.to_csv(
    "outputs/route_price_audit.csv",
    index=False
)

booking_stats.to_csv(
    "outputs/booking_window_audit.csv",
    index=False
)

airline_class_stats.to_csv(
    "outputs/airline_class_audit.csv",
    index=False
)

print("\nAudit reports saved in outputs/")
print("\nDATASET AUDIT COMPLETED")
