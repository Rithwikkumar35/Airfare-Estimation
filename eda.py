
# ============================================================
# INT395 - Airfare Component Price Estimation
# Step 4: Exploratory Data Analysis
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# 1. Paths and settings
# ------------------------------------------------------------

DATA_PATH = Path("data/Clean_Dataset.csv")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# 3. Chart 1: Airfare Price Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="price",
    bins=50,
    kde=True
)

plt.title("Distribution of Flight Ticket Prices")
plt.xlabel("Ticket Price (INR)")
plt.ylabel("Number of Flights")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "price_distribution.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 4. Chart 2: Average Price by Airline
# ------------------------------------------------------------

airline_prices = (
    df.groupby("airline")["price"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=airline_prices,
    x="airline",
    y="price"
)

plt.title("Average Flight Price by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Price (INR)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "airline_average_price.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 5. Chart 3: Average Price by Class
# ------------------------------------------------------------

class_prices = (
    df.groupby("class")["price"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=class_prices,
    x="class",
    y="price"
)

plt.title("Average Flight Price by Class")
plt.xlabel("Travel Class")
plt.ylabel("Average Price (INR)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_average_price.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 6. Chart 4: Days Before Departure vs Price
# ------------------------------------------------------------

# Sample data only for visualization.
# The full dataset remains unchanged.

plot_sample = df.sample(
    n=min(10000, len(df)),
    random_state=42
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=plot_sample,
    x="days_left",
    y="price",
    alpha=0.4
)

plt.title("Days Before Departure vs Flight Price")
plt.xlabel("Days Before Departure")
plt.ylabel("Ticket Price (INR)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "days_left_vs_price.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 7. Chart 5: Numerical Feature Correlation
# ------------------------------------------------------------

numerical_columns = [
    "duration",
    "days_left",
    "price"
]

correlation = df[numerical_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Between Numerical Features")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 8. Save EDA summary
# ------------------------------------------------------------

print("\nAVERAGE PRICE BY AIRLINE")
print(airline_prices)

print("\nAVERAGE PRICE BY CLASS")
print(class_prices)

print("\nNUMERICAL CORRELATION")
print(correlation)

print("\nAll EDA charts saved in the outputs folder!")

# ------------------------------------------------------------
# 9. Completion
# ------------------------------------------------------------

print("\nEDA COMPLETED SUCCESSFULLY!")
