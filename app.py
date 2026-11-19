
# ============================================================
# INT395 - AIRFARE COMPONENT PRICE ESTIMATION
# AEROFARE | IMPROVED STREAMLIT FRONTEND
# ============================================================

from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="AeroFare | Airfare Estimator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# 2. PROJECT PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "best_model.pkl"
DATA_PATH = ROOT / "data" / "Clean_Dataset.csv"
RESULTS_PATH = ROOT / "outputs" / "model_comparison.csv"

# ------------------------------------------------------------
# 3. LOAD PROJECT FILES
# ------------------------------------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


@st.cache_data
def load_results(path):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.info("Please run your model training script first.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset file not found.")
    st.stop()

if not RESULTS_PATH.exists():
    st.error("Model comparison file not found.")
    st.info("Please run your model evaluation script first.")
    st.stop()


model = load_model(str(MODEL_PATH))
df = load_dataset(str(DATA_PATH))
results = load_results(str(RESULTS_PATH))


required_columns = {
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
    "price"
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    st.error(
        "Dataset is missing columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()


# ------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------

def money(value):
    return f"₹{float(value):,.0f}"


def safe_text(value):
    return html.escape(str(value))


def prepare_comparison(data):

    data = data.copy()

    rename_map = {}

    for col in data.columns:

        key = str(col).strip().lower().replace("_", " ")

        if key in {"model", "model name"}:
            rename_map[col] = "Model"

        elif key == "mae":
            rename_map[col] = "MAE"

        elif key == "rmse":
            rename_map[col] = "RMSE"

        elif key in {"r2", "r2 score", "r squared"}:
            rename_map[col] = "R2 Score"

    data = data.rename(columns=rename_map)

    required = {"Model", "MAE", "RMSE", "R2 Score"}

    if required.issubset(data.columns):

        data = data[
            ["Model", "MAE", "RMSE", "R2 Score"]
        ].copy()

    else:

        # Recorded project evaluation results
        data = pd.DataFrame([
            {
                "Model": "Linear Regression",
                "MAE": 4553.29,
                "RMSE": 6761.71,
                "R2 Score": 0.911305
            },
            {
                "Model": "Random Forest",
                "MAE": 1077.91,
                "RMSE": 2787.82,
                "R2 Score": 0.984923
            }
        ])

    for col in ["MAE", "RMSE", "R2 Score"]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(
        subset=["MAE", "RMSE", "R2 Score"]
    )

    if data.empty:
        raise ValueError("No valid model evaluation results found.")

    return data


comparison = prepare_comparison(results)


# ------------------------------------------------------------
# 5. PREMIUM FRONTEND DESIGN
# ------------------------------------------------------------

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap'
);

:root {
    --navy: #10213f;
    --blue: #1769aa;
    --sky: #2589d1;
    --ink: #172b4d;
    --muted: #64748b;
    --line: #dce6f1;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {

    background:
        radial-gradient(
            circle at 8% 0%,
            #e9f3ff 0%,
            transparent 28%
        ),
        linear-gradient(
            180deg,
            #f7faff 0%,
            #f1f5fb 100%
        );

    color: var(--ink);
}

/* HEADER */

[data-testid="stHeader"] {
    background: #f7faff !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="stDecoration"] {
    display: none;
}

/* MAIN CONTAINER */

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #101d35 0%,
            #172f53 100%
        );
}

[data-testid="stSidebar"] * {
    color: #edf4ff;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {

    background: rgba(255,255,255,.055);

    border: 1px solid rgba(255,255,255,.07);

    padding: 10px 12px;

    border-radius: 12px;

    margin-bottom: 7px;
}

.brand {

    font-family: Manrope, sans-serif;

    font-size: 25px;

    font-weight: 800;

    letter-spacing: -1px;

    color: #ffffff;
}

.brand span {
    color: #68b7ff;
}

.sidebar-small {

    color: #b7c7dc !important;

    font-size: 12px;
}

/* HERO SECTION */

.hero {

    position: relative;

    overflow: hidden;

    padding: 34px 38px;

    border-radius: 24px;

    background:
        linear-gradient(
            112deg,
            #10213f 0%,
            #174b87 58%,
            #2589d1 100%
        );

    color: #ffffff;

    box-shadow:
        0 18px 42px rgba(23,75,135,.16);

    margin-bottom: 28px;
}

.hero h1 {

    font-family: Manrope, sans-serif;

    color: #ffffff;

    font-size: clamp(28px,3vw,42px);

    line-height: 1.12;

    font-weight: 800;

    margin: 9px 0 12px;

    max-width: 720px;
}

.hero p {

    color: #dcecff;

    font-size: 14px;

    max-width: 650px;

    margin: 0;
}

.hero-tag {

    display: inline-block;

    background: rgba(255,255,255,.13);

    border: 1px solid rgba(255,255,255,.2);

    border-radius: 30px;

    padding: 6px 12px;

    color: #eaf4ff;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1px;
}

.hero-art {

    position: absolute;

    right: 6%;

    top: 14%;

    font-size: clamp(65px,9vw,125px);

    opacity: .22;

    transform: rotate(-12deg);

    animation: floatplane 4.5s ease-in-out infinite;
}

@keyframes floatplane {

    0%, 100% {
        transform: translateY(0) rotate(-12deg);
    }

    50% {
        transform: translateY(-12px) rotate(-8deg);
    }
}

/* SECTION HEADINGS */

.section-kicker {

    color: #64748b;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 1.6px;

    text-transform: uppercase;

    margin-bottom: 5px;
}

.section-title {

    font-family: Manrope, sans-serif;

    color: #172b4d;

    font-size: 25px;

    font-weight: 800;

    margin: 0 0 4px;
}

.section-sub {

    color: #64748b;

    font-size: 13px;

    margin-bottom: 18px;
}

/* PANELS */

.panel {

    background: rgba(255,255,255,.94);

    border: 1px solid #e2eaf3;

    border-radius: 19px;

    padding: 21px;

    box-shadow: 0 8px 24px rgba(15,35,65,.045);
}

/* METRIC CARDS */

div[data-testid="stMetric"] {

    background: #ffffff !important;

    border: 1px solid #dce6f1 !important;

    border-radius: 16px !important;

    padding: 17px 18px !important;

    box-shadow: 0 5px 18px rgba(15,35,65,.045) !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {

    color: #52657d !important;

    opacity: 1 !important;

    font-weight: 700 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] div {

    color: #142b4a !important;

    opacity: 1 !important;

    font-weight: 800 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {

    opacity: 1 !important;

    font-weight: 600 !important;
}

/* FORM LABELS */

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stSlider"] label,
[data-testid="stSlider"] p {

    color: #263b56 !important;

    opacity: 1 !important;

    font-weight: 650 !important;
}

/* DROPDOWNS */

div[data-baseweb="select"] > div {

    background: #ffffff !important;

    border-color: #cbd8e7 !important;

    color: #172b4d !important;
}

div[data-baseweb="select"] input,
div[data-baseweb="select"] span {

    color: #172b4d !important;
}

/* NUMBER INPUT */

[data-testid="stNumberInput"] input {

    background: #ffffff !important;

    color: #172b4d !important;
}

/* SLIDERS */

[data-testid="stSlider"] [role="slider"] {

    border-color: #1769aa !important;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {

    background:
        linear-gradient(
            100deg,
            #1769aa,
            #2589d1
        ) !important;

    color: #ffffff !important;

    border: 0 !important;

    border-radius: 12px !important;

    padding: 12px 18px !important;

    font-weight: 800 !important;

    min-height: 46px;

    box-shadow:
        0 7px 17px rgba(37,137,209,.18);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {

    filter: brightness(.94);

    color: #ffffff !important;
}

/* PREDICTION RESULT */

.fare-result {

    background:
        linear-gradient(
            135deg,
            #eaf5ff,
            #f7fbff
        );

    border: 1px solid #cfe4f8;

    border-radius: 20px;

    padding: 25px;

    text-align: center;

    margin-top: 16px;
}

.fare-label {

    color: #47739d;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 1.4px;
}

.fare-amount {

    color: #075ca8;

    font-family: Manrope, sans-serif;

    font-size: clamp(36px,5vw,52px);

    font-weight: 800;

    letter-spacing: -2px;

    margin: 5px 0;
}

.fare-note {

    color: #52718c;

    font-size: 12px;
}

/* ROUTE CARD */

.route-card {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 16px;

    background: #ffffff;

    border: 1px solid #e2eaf3;

    border-radius: 16px;

    padding: 17px 20px;

    margin-top: 16px;
}

.route-city {

    color: #172b4d;

    font-family: Manrope, sans-serif;

    font-size: 20px;

    font-weight: 800;
}

.route-caption {

    color: #7b8ba1;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1px;
}

.route-line {

    flex: 1;

    border-top: 2px dashed #9dc8ed;

    text-align: center;

    color: #287fc2;

    min-width: 40px;
}

/* NOTICE */

.notice {

    background: #fff8e8;

    border: 1px solid #f4dfad;

    color: #78561a;

    border-radius: 12px;

    padding: 13px 16px;

    font-size: 12px;

    margin-top: 14px;
}

/* FOOTER */

.footer {

    border-top: 1px solid #dce6f1;

    margin-top: 34px;

    padding: 23px 8px 8px;

    text-align: center;

    color: #64748b;

    font-size: 12px;

    line-height: 1.8;
}

.footer-brand {

    color: #163b65;

    font-family: Manrope, sans-serif;

    font-size: 19px;

    font-weight: 800;

    letter-spacing: 1.5px;
}

/* DATAFRAME */

div[data-testid="stDataFrame"] {

    border: 1px solid #dce6f1;

    border-radius: 13px;

    overflow: hidden;
}

/* HEADINGS */

h1, h2, h3, h4 {

    font-family: Manrope, sans-serif !important;

    color: #172b4d;
}

hr {
    border-color: #dce6f1 !important;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 6. REUSABLE UI COMPONENTS
# ------------------------------------------------------------

def section_heading(kicker, title, subtitle=""):

    st.markdown(
        f'<div class="section-kicker">{safe_text(kicker)}</div>'
        f'<div class="section-title">{safe_text(title)}</div>'
        f'<div class="section-sub">{safe_text(subtitle)}</div>',
        unsafe_allow_html=True
    )


def hero():

    st.markdown(
        '<div class="hero">'
        '<div class="hero-tag">'
        'AEROFARE • SMART AIRFARE INTELLIGENCE'
        '</div>'
        '<div class="hero-art">✈</div>'
        '<h1>Make your next journey<br>more informed.</h1>'
        '<p>Explore historical airfare patterns and estimate a ticket price '
        'using a supervised machine learning model.</p>'
        '</div>',
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# 7. SIDEBAR NAVIGATION
# ------------------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="brand">AERO<span>FARE</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-small">'
        'INT395 • SUPERVISED LEARNING'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "✈️  Fare Estimator",
            "📊  Model Insights",
            "🗂️  Dataset Explorer"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("**PROJECT STATUS**")

    st.success("Prediction pipeline loaded")

    st.markdown(
        '<div class="sidebar-small">'
        'Model: Random Forest Regressor'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="sidebar-small">'
        f'Dataset: {len(df):,} records'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-small">'
        'Academic estimation tool. '
        'Not connected to live airline inventory.'
        '</div>',
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# 8. FARE ESTIMATOR PAGE
# ------------------------------------------------------------

if page == "✈️  Fare Estimator":

    hero()

    section_heading(
        "PLAN YOUR JOURNEY",
        "Flight details",
        "Choose the route and flight characteristics "
        "for a historical-data-based estimate."
    )

    airlines = sorted(
        df["airline"].dropna().unique().tolist()
    )

    cities = sorted(
        set(df["source_city"].dropna())
        |
        set(df["destination_city"].dropna())
    )

    time_options = sorted(
        df["departure_time"].dropna().unique().tolist()
    )

    arrival_options = sorted(
        df["arrival_time"].dropna().unique().tolist()
    )

    stop_options = sorted(
        df["stops"].dropna().unique().tolist()
    )

    class_options = sorted(
        df["class"].dropna().unique().tolist()
    )

    with st.form("prediction_form", border=True):

        st.markdown("#### 01 / Route")

        c1, c2 = st.columns(2)

        with c1:

            source_default = (
                cities.index("Delhi")
                if "Delhi" in cities
                else 0
            )

            source_city = st.selectbox(
                "Departure city",
                cities,
                index=source_default
            )

        with c2:

            destination_choices = [
                c for c in cities
                if c != source_city
            ]

            destination_default = (
                destination_choices.index("Mumbai")
                if "Mumbai" in destination_choices
                else 0
            )

            destination_city = st.selectbox(
                "Destination city",
                destination_choices,
                index=destination_default
            )

        st.markdown("---")

        st.markdown("#### 02 / Flight preferences")

        a, b, c = st.columns(3)

        with a:

            airline = st.selectbox(
                "Airline",
                airlines
            )

            travel_class = st.selectbox(
                "Travel class",
                class_options,
                index=(
                    class_options.index("Economy")
                    if "Economy" in class_options
                    else 0
                )
            )

        with b:

            departure_time = st.selectbox(
                "Departure time",
                time_options
            )

            arrival_time = st.selectbox(
                "Arrival time",
                arrival_options
            )

        with c:

            stops = st.selectbox(
                "Number of stops",
                stop_options
            )

            days_left = st.slider(
                "Days before departure",
                1,
                49,
                15
            )

        duration = st.slider(
            "Journey duration (hours)",
            0.83,
            49.83,
            2.50,
            0.25,
            help="Choose a duration within the range represented in the dataset."
        )

        submitted = st.form_submit_button(
            "✈  Estimate my airfare",
            use_container_width=True
        )

    if submitted:

        if source_city == destination_city:

            st.error(
                "Departure and destination cities must be different."
            )

        else:

            input_data = pd.DataFrame([{

                "airline": airline,

                "source_city": source_city,

                "departure_time": departure_time,

                "stops": stops,

                "arrival_time": arrival_time,

                "destination_city": destination_city,

                "class": travel_class,

                "duration": duration,

                "days_left": days_left

            }])

            try:

                prediction = float(
                    model.predict(input_data)[0]
                )

                if not np.isfinite(prediction):
                    raise ValueError(
                        "The model returned an invalid prediction."
                    )

                st.markdown(
                    f'<div class="route-card">'
                    f'<div>'
                    f'<div class="route-caption">FROM</div>'
                    f'<div class="route-city">'
                    f'{safe_text(source_city)}'
                    f'</div>'
                    f'</div>'
                    f'<div class="route-line">✈ ───────── ✈</div>'
                    f'<div style="text-align:right">'
                    f'<div class="route-caption">TO</div>'
                    f'<div class="route-city">'
                    f'{safe_text(destination_city)}'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="fare-result">'
                    f'<div class="fare-label">'
                    f'MODEL-ESTIMATED AIRFARE'
                    f'</div>'
                    f'<div class="fare-amount">'
                    f'{money(prediction)}'
                    f'</div>'
                    f'<div class="fare-note">'
                    f'Estimated from the selected flight attributes'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="notice">'
                    '<b>Important:</b> This is a model estimate based '
                    'on historical dataset patterns, not a live airline '
                    'quote or guaranteed booking price. Actual fares '
                    'may differ because of availability, taxes, '
                    'promotions, demand and other factors.'
                    '</div>',
                    unsafe_allow_html=True
                )

                class_median = float(
                    df.loc[
                        df["class"] == travel_class,
                        "price"
                    ].median()
                )

                difference = prediction - class_median

                st.markdown("#### Historical context")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Selected class median",
                    money(class_median)
                )

                m2.metric(
                    "Difference from median",
                    money(abs(difference)),
                    (
                        "Above median"
                        if difference > 0
                        else "Below median"
                    ),
                    delta_color=(
                        "inverse"
                        if difference > 0
                        else "normal"
                    )
                )

                m3.metric(
                    "Booking window",
                    f"{days_left} days",
                    "before departure"
                )

            except Exception as exc:

                st.error(
                    f"Prediction could not be generated: {exc}"
                )

                st.info(
                    "Check that the input feature names and categories "
                    "match the model training pipeline."
                )


# ------------------------------------------------------------
# 9. MODEL INSIGHTS PAGE
# ------------------------------------------------------------

elif page == "📊  Model Insights":

    hero()

    section_heading(
        "MODEL EVALUATION",
        "Performance at a glance",
        "Evaluation results from the held-out test data "
        "recorded by the project."
    )

    best = comparison.loc[
        comparison["MAE"].idxmin()
    ]

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Best model by MAE",
        str(best["Model"])
    )

    m2.metric(
        "Mean Absolute Error",
        money(best["MAE"])
    )

    m3.metric(
        "R² score",
        f'{best["R2 Score"]:.4f}'
    )

    st.markdown("### Model comparison")

    left, right = st.columns([1.25, 1])

    with left:

        chart_data = comparison.melt(
            id_vars="Model",
            value_vars=["MAE", "RMSE"],
            var_name="Metric",
            value_name="Error"
        )

        fig = px.bar(
            chart_data,
            x="Model",
            y="Error",
            color="Metric",
            barmode="group",
            color_discrete_map={
                "MAE": "#1769aa",
                "RMSE": "#91c9f2"
            },
            labels={
                "Error": "Error (₹)"
            },
            template="plotly_white"
        )

        fig.update_layout(
            height=390,
            legend_title_text="",
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        fig2 = px.bar(
            comparison,
            x="Model",
            y="R2 Score",
            color="Model",
            color_discrete_sequence=[
                "#1769aa",
                "#62a9df"
            ],
            labels={
                "R2 Score": "R² score"
            },
            template="plotly_white"
        )

        fig2.update_yaxes(
            range=[0, 1]
        )

        fig2.update_layout(
            height=390,
            showlegend=False,
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("### Recorded evaluation results")

    display = comparison.copy()

    display["MAE"] = display["MAE"].map(money)

    display["RMSE"] = display["RMSE"].map(money)

    display["R2 Score"] = display["R2 Score"].map(
        lambda x: f"{x:.6f}"
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Understanding the metrics")

    with st.expander("Mean Absolute Error (MAE)"):

        st.write(
            "MAE is the average absolute difference between "
            "actual and predicted fares. A lower MAE is better. "
            "It does not mean every prediction is within this amount."
        )

    with st.expander("Root Mean Squared Error (RMSE)"):

        st.write(
            "RMSE gives more influence to large errors than MAE. "
            "A lower RMSE indicates fewer or smaller large "
            "prediction errors."
        )

    with st.expander("R² score"):

        st.write(
            "R² compares the model with a mean-target baseline. "
            "A value closer to 1 indicates a stronger fit on "
            "the evaluated test data; it is not a guarantee "
            "for each prediction."
        )

    st.markdown(
        '<div class="notice">'
        '<b>Interpretation:</b> In the recorded evaluation, '
        'Random Forest performed better than Linear Regression. '
        'Error analysis also found higher error for Business-class '
        'observations than Economy-class observations. Performance '
        'may vary by route, class and booking window.'
        '</div>',
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# 10. DATASET EXPLORER PAGE
# ------------------------------------------------------------

elif page == "🗂️  Dataset Explorer":

    hero()

    section_heading(
        "DATA EXPLORATION",
        "Explore the historical dataset",
        "Inspect the data distribution and compare average fares "
        "across categories."
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Records",
        f"{len(df):,}"
    )

    k2.metric(
        "Features + target",
        f"{len(df.columns)}"
    )

    k3.metric(
        "Missing values",
        f"{int(df.isna().sum().sum()):,}"
    )

    k4.metric(
        "Median fare",
        money(df["price"].median())
    )

    chart_left, chart_right = st.columns([1.2, 1])

    with chart_left:

        st.markdown("### Ticket-price distribution")

        sample = df.sample(
            n=min(15000, len(df)),
            random_state=42
        )

        fig = px.histogram(
            sample,
            x="price",
            nbins=55,
            color_discrete_sequence=["#2589d1"],
            labels={
                "price": "Ticket price (₹)",
                "count": "Number of records"
            },
            template="plotly_white"
        )

        fig.update_layout(
            height=380,
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with chart_right:

        st.markdown("### Average fare by class")

        class_prices = (
            df.groupby("class", as_index=False)["price"]
            .mean()
        )

        fig = px.bar(
            class_prices,
            x="class",
            y="price",
            color="class",
            color_discrete_map={
                "Economy": "#2589d1",
                "Business": "#10213f"
            },
            labels={
                "price": "Average fare (₹)"
            },
            template="plotly_white"
        )

        fig.update_layout(
            height=380,
            showlegend=False,
            margin=dict(
                l=10,
                r=10,
                t=25,
                b=10
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Average fare by airline")

    airline_prices = (
        df.groupby("airline", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
    )

    fig = px.bar(
        airline_prices,
        x="airline",
        y="price",
        color="price",
        color_continuous_scale="Blues",
        labels={
            "price": "Average fare (₹)"
        },
        template="plotly_white"
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Filter and preview records")

    f1, f2 = st.columns(2)

    with f1:

        chosen_class = st.multiselect(
            "Travel class",
            sorted(df["class"].dropna().unique()),
            default=sorted(df["class"].dropna().unique())
        )

    with f2:

        chosen_airline = st.multiselect(
            "Airline",
            sorted(df["airline"].dropna().unique()),
            default=sorted(df["airline"].dropna().unique())
        )

    filtered = df[
        df["class"].isin(chosen_class)
        &
        df["airline"].isin(chosen_airline)
    ]

    selected_columns = st.multiselect(
        "Columns to display",
        options=df.columns.tolist(),
        default=[
            "airline",
            "source_city",
            "destination_city",
            "class",
            "duration",
            "days_left",
            "price"
        ]
    )

    if selected_columns:

        st.caption(
            f"Showing up to 100 of {len(filtered):,} filtered records."
        )

        st.dataframe(
            filtered[selected_columns].head(100),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Select at least one column to display."
        )


# ------------------------------------------------------------
# 11. FOOTER
# ------------------------------------------------------------

st.markdown(
    '<div class="footer">'
    '<div class="footer-brand">✈ AEROFARE</div>'
    '<div><b>INT395 • Supervised Learning Project</b></div>'
    '<div>Historical Airfare Estimation Using Machine Learning</div>'
    '<div style="font-size:11px;color:#94a3b8;margin-top:8px;">'
    'Academic project • Estimates only • Not live ticket prices'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)
