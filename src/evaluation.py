"""
AeroFare - Interactive Visualization & Metric Evaluation Module
Generates Plotly charts for model evaluation, residuals, and diagnostic analytics.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

NAVY_COLOR = "#0A192F"
BLUE_COLOR = "#1769AA"
ACCENT_CYAN = "#00B4D8"
CORAL_ACCENT = "#E63946"
SLATE_GRAY = "#64748B"


def create_model_comparison_bar(df_comparison: pd.DataFrame) -> go.Figure:
    """Creates a grouped bar chart comparing MAE and RMSE across all models."""
    df_plot = df_comparison.copy()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_plot["Model"],
        y=df_plot["MAE"],
        name="MAE (₹)",
        marker_color="#1769AA",
        text=df_plot["MAE"].apply(lambda v: f"₹{v:,.0f}"),
        textposition="auto"
    ))

    fig.add_trace(go.Bar(
        x=df_plot["Model"],
        y=df_plot["RMSE"],
        name="RMSE (₹)",
        marker_color="#00B4D8",
        text=df_plot["RMSE"].apply(lambda v: f"₹{v:,.0f}"),
        textposition="auto"
    ))

    fig.update_layout(
        title="Model Error Metrics Comparison (Lower is Better)",
        xaxis_title="Evaluated Model Architecture",
        yaxis_title="Error (INR ₹)",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=30),
        height=420
    )
    return fig


def create_feature_importance_chart(df_importance: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Creates a horizontal bar chart of top feature importances."""
    df_top = df_importance.head(top_n).sort_values(by="Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_top["Importance"],
        y=df_top["Feature"],
        orientation="h",
        marker=dict(
            color=df_top["Importance"],
            colorscale="Blues",
            showscale=False
        ),
        text=df_top["Importance"].apply(lambda v: f"{v*100:.1f}%"),
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Top {top_n} Predictive Feature Importances",
        xaxis_title="Relative Feature Importance Score",
        yaxis_title="Feature",
        template="plotly_white",
        margin=dict(l=20, r=40, t=40, b=20),
        height=380
    )
    return fig


def create_class_error_chart(df_class: pd.DataFrame) -> go.Figure:
    """Creates a comparison of MAE and RMSE across Economy and Business classes."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_class["class"],
        y=df_class["MAE"],
        name="MAE",
        marker_color="#1769AA",
        text=df_class["MAE"].apply(lambda v: f"₹{v:,.0f}"),
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=df_class["class"],
        y=df_class["RMSE"],
        name="RMSE",
        marker_color="#E63946",
        text=df_class["RMSE"].apply(lambda v: f"₹{v:,.0f}"),
        textposition="auto"
    ))

    fig.update_layout(
        title="Error Disparity by Travel Class",
        xaxis_title="Travel Class",
        yaxis_title="Error (INR ₹)",
        barmode="group",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig


def create_booking_window_chart(df_window: pd.DataFrame) -> go.Figure:
    """Plots MAE and Median Price trends across booking windows."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_window["booking_window"],
        y=df_window["MAE"],
        name="Mean Absolute Error (₹)",
        marker_color="#1769AA",
        text=df_window["MAE"].apply(lambda v: f"₹{v:,.0f}"),
        textposition="auto",
        yaxis="y1"
    ))

    if "median_actual" in df_window.columns:
        fig.add_trace(go.Scatter(
            x=df_window["booking_window"],
            y=df_window["median_actual"],
            name="Median Ticket Price (₹)",
            mode="lines+markers",
            line=dict(color="#E63946", width=3),
            yaxis="y2"
        ))

    fig.update_layout(
        title="Fare Estimation Error vs Advance Booking Window",
        xaxis_title="Days Remaining Before Departure",
        yaxis=dict(title="MAE (₹)", showgrid=True),
        yaxis2=dict(
            title="Median Ticket Price (₹)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=20, r=40, t=60, b=20),
        height=400
    )
    return fig


def create_actual_vs_predicted_scatter(df_diagnostics: pd.DataFrame, sample_size: int = 3000) -> go.Figure:
    """Scatter plot of actual vs predicted ticket prices with 45-degree reference line."""
    if len(df_diagnostics) > sample_size:
        sample_df = df_diagnostics.sample(n=sample_size, random_state=42)
    else:
        sample_df = df_diagnostics

    fig = go.Figure()

    # Perfect prediction line
    max_val = max(sample_df["actual_price"].max(), sample_df["predicted_price"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Perfect Fit (y = x)",
        line=dict(color="#64748B", dash="dash", width=2)
    ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=sample_df["actual_price"],
        y=sample_df["predicted_price"],
        mode="markers",
        name="Test Sample Flights",
        marker=dict(
            color=sample_df["class"].map({"Economy": "#1769AA", "Business": "#00B4D8"}),
            opacity=0.5,
            size=5
        )
    ))

    fig.update_layout(
        title=f"Actual vs. Predicted Flight Prices (Sample: {len(sample_df):,} test records)",
        xaxis_title="Actual Flight Price (INR ₹)",
        yaxis_title="Model Predicted Price (INR ₹)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=420
    )
    return fig


def create_residual_distribution(df_diagnostics: pd.DataFrame, sample_size: int = 5000) -> go.Figure:
    """Plots histogram of model residuals (Predicted - Actual)."""
    if len(df_diagnostics) > sample_size:
        sample_df = df_diagnostics.sample(n=sample_size, random_state=42)
    else:
        sample_df = df_diagnostics

    fig = px.histogram(
        sample_df,
        x="error",
        color="class",
        nbins=60,
        marginal="box",
        title="Model Residual Distribution by Travel Class (Prediction - Actual)",
        labels={"error": "Residual Error (INR ₹)", "class": "Travel Class"},
        color_discrete_map={"Economy": "#1769AA", "Business": "#00B4D8"},
        template="plotly_white"
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    return fig
