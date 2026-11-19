# ============================================================
# INT395 - Airfare Component Price Estimation
# Step 8.2: Model Error Diagnostics
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. Load dataset and trained model
# ------------------------------------------------------------

df = pd.read_csv("data/Clean_Dataset.csv")

# Match the preprocessing used during model training
df = df.drop(columns=["flight"])

X = df.drop(columns=["price"])
y = df["price"]

model = joblib.load("models/best_model.pkl")

# ------------------------------------------------------------
# 2. Recreate the same train/test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("=" * 65)
print("INT395 AIRFARE MODEL DIAGNOSTICS")
print("=" * 65)

# ------------------------------------------------------------
# 3. Predict test-set prices
# ------------------------------------------------------------

predictions = model.predict(X_test)

results = X_test.copy()
results["actual_price"] = y_test.values
results["predicted_price"] = predictions
results["error"] = results["predicted_price"] - results["actual_price"]
results["absolute_error"] = results["error"].abs()

# ------------------------------------------------------------
# 4. Overall performance
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n1. OVERALL TEST PERFORMANCE")
print(f"MAE:  ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R²:   {r2:.5f}")

# ------------------------------------------------------------
# 5. Performance by travel class
# ------------------------------------------------------------

print("\n2. PERFORMANCE BY CLASS")

class_results = []

for travel_class, group in results.groupby("class"):
    class_mae = mean_absolute_error(
        group["actual_price"],
        group["predicted_price"]
    )

    class_rmse = mean_squared_error(
        group["actual_price"],
        group["predicted_price"]
    ) ** 0.5

    class_results.append({
        "class": travel_class,
        "count": len(group),
        "MAE": class_mae,
        "RMSE": class_rmse,
        "median_actual": group["actual_price"].median(),
        "median_predicted": group["predicted_price"].median()
    })

class_report = pd.DataFrame(class_results)
print(class_report.to_string(index=False))

# ------------------------------------------------------------
# 6. Performance by booking window
# ------------------------------------------------------------

print("\n3. PERFORMANCE BY DAYS LEFT")

results["booking_window"] = pd.cut(
    results["days_left"],
    bins=[0, 7, 14, 21, 30, 40, 50],
    labels=[
        "1–7 days",
        "8–14 days",
        "15–21 days",
        "22–30 days",
        "31–40 days",
        "41–49 days"
    ],
    include_lowest=True
)

window_report = (
    results.groupby("booking_window", observed=False)
    .agg(
        count=("actual_price", "size"),
        MAE=("absolute_error", "mean"),
        median_actual=("actual_price", "median"),
        median_predicted=("predicted_price", "median")
    )
    .reset_index()
)

print(window_report.to_string(index=False))

# ------------------------------------------------------------
# 7. Largest overpredictions
# ------------------------------------------------------------

print("\n4. TOP 10 OVERPREDICTIONS")
overpredictions = results.sort_values(
    "error",
    ascending=False
).head(10)

print(
    overpredictions[
        [
            "airline",
            "source_city",
            "destination_city",
            "class",
            "days_left",
            "actual_price",
            "predicted_price",
            "error"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 8. Largest underpredictions
# ------------------------------------------------------------

print("\n5. TOP 10 UNDERPREDICTIONS")
underpredictions = results.sort_values(
    "error",
    ascending=True
).head(10)

print(
    underpredictions[
        [
            "airline",
            "source_city",
            "destination_city",
            "class",
            "days_left",
            "actual_price",
            "predicted_price",
            "error"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 9. Save diagnostic reports
# ------------------------------------------------------------

results.to_csv("outputs/test_predictions_diagnostics.csv", index=False)
class_report.to_csv("outputs/class_performance.csv", index=False)
window_report.to_csv("outputs/booking_window_performance.csv", index=False)

# Consolidated diagnostics report
consolidated = []
consolidated.append({
    "Category": "Overall Test Set",
    "Subcategory": "All Flights",
    "Count": len(results),
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2": round(r2, 4)
})

for _, row in class_report.iterrows():
    consolidated.append({
        "Category": "Travel Class",
        "Subcategory": row["class"],
        "Count": int(row["count"]),
        "MAE": round(row["MAE"], 2),
        "RMSE": round(row["RMSE"], 2),
        "R2": "N/A"
    })

for _, row in window_report.iterrows():
    consolidated.append({
        "Category": "Booking Window",
        "Subcategory": str(row["booking_window"]),
        "Count": int(row["count"]),
        "MAE": round(row["MAE"], 2),
        "RMSE": "N/A",
        "R2": "N/A"
    })

consolidated_df = pd.DataFrame(consolidated)
consolidated_df.to_csv("outputs/model_diagnostics.csv", index=False)

print("\nReports saved in outputs/ including outputs/model_diagnostics.csv")
print("MODEL DIAGNOSTICS COMPLETED")