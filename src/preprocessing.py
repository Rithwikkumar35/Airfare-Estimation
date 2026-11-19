"""
AeroFare - Preprocessing & Input Validation Module
Ensures rigorous schema integrity between UI input and scikit-learn pipeline.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd

AIRLINES = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
DEPARTURE_TIMES = ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"]
ARRIVAL_TIMES = ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"]
STOPS = ["zero", "one", "two_or_more"]
CLASSES = ["Economy", "Business"]

NUMERICAL_FEATURES = ["duration", "days_left"]
CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class"
]
FEATURE_ORDER = [
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


def validate_flight_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates user input against physical and domain constraints.
    Returns (is_valid, list_of_error_messages).
    """
    errors = []

    # 1. Required keys presence
    for feat in FEATURE_ORDER:
        if feat not in payload:
            errors.append(f"Missing required parameter: '{feat}'")

    if errors:
        return False, errors

    # 2. City Route Logic: Source and destination must not be the same
    src = payload.get("source_city")
    dst = payload.get("destination_city")
    if src == dst:
        errors.append(f"Source city and destination city cannot be identical ('{src}'). Domestic flights require distinct origin and arrival cities.")

    # 3. Categorical Domain Validation
    if payload.get("airline") not in AIRLINES:
        errors.append(f"Invalid airline: '{payload.get('airline')}'. Must be one of: {AIRLINES}")
    if src not in CITIES:
        errors.append(f"Invalid source city: '{src}'. Must be one of: {CITIES}")
    if dst not in CITIES:
        errors.append(f"Invalid destination city: '{dst}'. Must be one of: {CITIES}")
    if payload.get("departure_time") not in DEPARTURE_TIMES:
        errors.append(f"Invalid departure time: '{payload.get('departure_time')}'")
    if payload.get("arrival_time") not in ARRIVAL_TIMES:
        errors.append(f"Invalid arrival time: '{payload.get('arrival_time')}'")
    if payload.get("stops") not in STOPS:
        errors.append(f"Invalid stops: '{payload.get('stops')}'")
    if payload.get("class") not in CLASSES:
        errors.append(f"Invalid travel class: '{payload.get('class')}'")

    # 4. Numerical Range Validation
    try:
        duration = float(payload.get("duration", 0))
        if duration < 0.5 or duration > 50.0:
            errors.append(f"Flight duration ({duration:.2f} hrs) is outside reasonable domain [0.5, 50.0] hours.")
    except (ValueError, TypeError):
        errors.append("Duration must be a numerical value.")

    try:
        days_left = int(payload.get("days_left", 0))
        if days_left < 1 or days_left > 50:
            errors.append(f"Days left ({days_left}) is outside domain [1, 50] days.")
    except (ValueError, TypeError):
        errors.append("Days left must be an integer.")

    return len(errors) == 0, errors


def prepare_inference_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms payload into a single-row DataFrame ordered exactly as expected
    by the trained ColumnTransformer pipeline.
    """
    row_data = {
        "airline": [str(payload["airline"])],
        "source_city": [str(payload["source_city"])],
        "departure_time": [str(payload["departure_time"])],
        "stops": [str(payload["stops"])],
        "arrival_time": [str(payload["arrival_time"])],
        "destination_city": [str(payload["destination_city"])],
        "class": [str(payload["class"])],
        "duration": [float(payload["duration"])],
        "days_left": [int(payload["days_left"])]
    }
    df = pd.DataFrame(row_data)
    return df[FEATURE_ORDER]
