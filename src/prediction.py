"""
AeroFare - Inference Engine & Statistical Uncertainty Module
Executes predictions through the loaded pipeline and calculates statistically defensible error bands.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.preprocessing import validate_flight_payload, prepare_inference_df
from src.data_loader import load_model, load_model_metadata

MINIMUM_FARE_FLOOR = 1105.0  # Lowest historical domestic ticket price in dataset

# Empirical Test-Set RMSE values by class from model_diagnostics.csv
CLASS_TEST_RMSE = {
    "Economy": 1459.93,
    "Business": 4558.38
}

# Booking window surge uncertainty multipliers (based on window_report residuals)
WINDOW_UNCERTAINTY_MULTIPLIERS = {
    "1–7 days": 1.45,
    "8–14 days": 1.15,
    "15–21 days": 1.00,
    "22–30 days": 0.85,
    "31–40 days": 0.80,
    "41–49 days": 0.82
}


def get_booking_window_bin(days_left: int) -> str:
    """Classifies days remaining into empirical diagnostic bins."""
    if days_left <= 7:
        return "1–7 days"
    elif days_left <= 14:
        return "8–14 days"
    elif days_left <= 21:
        return "15–21 days"
    elif days_left <= 30:
        return "22–30 days"
    elif days_left <= 40:
        return "31–40 days"
    else:
        return "41–49 days"


def predict_fare(payload: Dict[str, Any], pipeline=None) -> Dict[str, Any]:
    """
    Validates payload, formats features, runs pipeline inference,
    and returns point estimate with statistical bounds.
    """
    is_valid, error_list = validate_flight_payload(payload)
    if not is_valid:
        raise ValueError("; ".join(error_list))

    if pipeline is None:
        pipeline = load_model()

    input_df = prepare_inference_df(payload)
    raw_pred = pipeline.predict(input_df)[0]
    estimated_fare = max(MINIMUM_FARE_FLOOR, float(raw_pred))

    # Calculate Statistically Defensible Uncertainty Range
    travel_class = payload["class"]
    days_left = int(payload["days_left"])
    window_bin = get_booking_window_bin(days_left)

    base_rmse = CLASS_TEST_RMSE.get(travel_class, 2818.66)
    window_multiplier = WINDOW_UNCERTAINTY_MULTIPLIERS.get(window_bin, 1.0)
    effective_sigma = base_rmse * window_multiplier

    # 90% Empirical Confidence Interval (z ~ 1.645)
    margin = 1.645 * effective_sigma
    lower_bound = max(MINIMUM_FARE_FLOOR, estimated_fare - margin)
    upper_bound = estimated_fare + margin

    metadata = load_model_metadata()

    return {
        "status": "success",
        "estimated_fare": round(estimated_fare, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "margin_of_error": round(margin, 2),
        "route": f"{payload['source_city']} → {payload['destination_city']}",
        "airline": payload["airline"],
        "class": travel_class,
        "days_left": days_left,
        "booking_window": window_bin,
        "duration": float(payload["duration"]),
        "stops": payload["stops"],
        "departure_time": payload["departure_time"],
        "arrival_time": payload["arrival_time"],
        "model_name": metadata.get("model_name", "Optimized Random Forest"),
        "model_r2": metadata.get("metrics", {}).get("r2_score", 0.9846),
        "model_mae": metadata.get("metrics", {}).get("mae", 1318.35),
        "uncertainty_explanation": (
            f"Range represents an empirical 90% prediction band based on observed test-set "
            f"residuals for {travel_class} class (Base RMSE ₹{base_rmse:,.0f}) and "
            f"booking window '{window_bin}' (adjustment factor {window_multiplier:.2f}x)."
        ),
        "disclaimer": "Estimated fare based on historical dataset patterns. This is not a live airline quote or guaranteed booking price."
    }
