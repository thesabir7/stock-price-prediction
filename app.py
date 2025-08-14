import io
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock 30-Day Forecaster", layout="wide")
st.title("📈 Stock Market Prediction App — Next 30 Days")
st.caption("Upload a historical price CSV → choose model → get 30‑day forecast with plots & downloadable CSV.")

# -----------------------------
# Helpers
# -----------------------------

def _coerce_to_datetime(series: pd.Series) -> pd.Series:
    """Best‑effort parse to datetime with day-first fallback."""
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    # Try day-first if many NaTs
    if s.isna().mean() > 0.2:
        s = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s

@st.cache_data(show_spinner=False)
def _read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def _prepare_ts(df: pd.DataFrame, date_col: str, value_col: str, freq: str) -> pd.DataFrame:
    df = df[[date_col, value_col]].copy()
    df[date_col] = _coerce_to_datetime(df[date_col])
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    # Keep numeric only
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    # Resample to target frequency and forward fill small gaps
    df = df.resample(freq).last()
    df[value_col] = df[value_col].ffill()
    df = df.dropna()
    return df


def _train_holt_winters(y: pd.Series, seasonal: Optional[str], seasonal_periods: Optional[int]):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    ).fit(optimized=True)
    return model


def _try_import_prophet():
    try:
        from prophet import Prophet  # type: ignore
        return Prophet
    except Exception as e:
        return None


def _fit_prophet(df: pd.DataFrame, value_col: str):
    Prophet = _try_import_prophet()
    if Prophet is None:
        raise ImportError(
            "The 'prophet' package is not installed. Add 'prophet' to requirements and try again."
        )
    tmp = df.reset_index().rename(columns={df.index.name: "ds", value_col: "y"})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(tmp)
    return m


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    eps = 1e-8
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    demo = st.toggle("Use demo data (AAPL daily close)", value=True)

    freq = st.selectbox(
        "Target frequency",
        options=["B", "D", "W", "M"],
        index=0,
        help="Resample data to this frequency before modeling: B=Business day, D=Calendar day, W=Weekly, M=Monthly",
    )

    model_choice = st.selectbox(
        "Model",
        options=["Holt‑Winters (ExponentialSmoothing)", "Prophet"],
        help="Prophet requires the 'prophet' package. Holt‑Winters is lightweight and robust.",
    )

    horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=30, step=1)

    test_size = st.slider("Hold‑out size (% of history)", min_value=5, max_value=30, value=15, step=1)


# -----------------------------
# Data input
# -----------------------------
if demo:
    rng = pd.date_range("2023-01-02", periods=420, freq="B")
    np.random.seed(7)
    trend = np.linspace(140, 200, len(rng))
    noise = np.random.normal(scale=2.0, size=len(rng))
    season = 2 * np.sin(np.arange(len(rng)) / 14)
    price = trend + season + noise
    df_raw = pd.DataFrame({"Date": rng, "Close": price})
    st.success("Loaded demo data: 420 business‑day rows of synthetic AAPL‑like close prices.")
    date_col_default, value_col_default = "Date", "Close"
else:
    uploaded = st.file_uploader("Upload CSV with at least a Date column and a price column", type=["csv"]) 
    if uploaded is None:
        st.info("Upload a CSV or enable demo data in the sidebar to proceed.")
        st.stop()
    df_raw = _read_csv(uploaded)
    date_col_default = next((c for c in df_raw.columns if c.lower() in {"date", "timestamp", "time"}), df_raw.columns[0])
    candidates = [c for c in df_raw.columns if c.lower() in {"close", "adj close", "price", "close_price"}]
    value_col_default = candidates[0] if candidates else df_raw.columns[-1]

st.subheader("1) Select columns")
col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Date column", options=list(df_raw.columns), index=list(df_raw.columns).index(date_col_default))
with col2:
    value_col = st.selectbox("Price/Target column", options=list(df_raw.columns), index=list(df_raw.columns).index(value_col_default))

# Prepare time series
try:
    df_ts = _prepare_ts(df_raw, date_col=date_col, value_col=value_col, freq=freq)
except Exception as e:
    st.error(f"Failed to parse/prepare your data: {e}")
    st.stop()

st.subheader("2) Preview & sanity checks")
st.write(df_ts.tail(10))

fig_hist = px.line(df_ts.reset_index(), x=df_ts.index.name, y=value_col, title="Historical series (resampled)")
st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------
# Train / validate
# -----------------------------
st.subheader("3) Train model & validate")

if len(df_ts) < 30:
    st.error("Need at least 30 rows after resampling to fit a reliable model.")
    st.stop()

n_test = max(1, int(len(df_ts) * (test_size / 100)))
train = df_ts.iloc[:-n_test]
test = df_ts.iloc[-n_test:]

if model_choice.startswith("Holt"):
    seasonal = None
    seasonal_periods = None
    if freq in ("B", "D") and len(train) >= 60:
        seasonal, seasonal_periods = "add", 5 if freq == "B" else 7
    elif freq == "W" and len(train) >= 120:
        seasonal, seasonal_periods = "add", 52
    elif freq == "M" and len(train) >= 48:
        seasonal, seasonal_periods = "add", 12
    model = _train_holt_winters(train[value_col], seasonal, seasonal_periods)
    fitted = model.fittedvalues.reindex(train.index)
    test_fc = model.forecast(steps=len(test))
    future_fc = model.forecast(steps=horizon)

elif model_choice == "Prophet":
    try:
        m = _fit_prophet(train, value_col)
        fitted_df = m.predict(train.reset_index().rename(columns={train.index.name: "ds"}))[["ds", "yhat"]]
        fitted = fitted_df.set_index("ds").iloc[:, 0].rename(value_col)
        future_test = m.make_future_dataframe(periods=len(test), freq=freq)
        test_pred = m.predict(future_test).set_index("ds").iloc[-len(test):]["yhat"]
        future_future = m.make_future_dataframe(periods=horizon, freq=freq)
        future_fc = m.predict(future_future).set_index("ds").iloc[-horizon:]["yhat"]
        test_fc = test_pred
    except ImportError as e:
        st.error(str(e))
        st.stop()
else:
    st.error("Unknown model choice.")
    st.stop()

# Metrics
met = _metrics(test[value_col].to_numpy(), pd.Series(test_fc, index=test.index).to_numpy())
colA, colB, colC = st.columns(3)
colA.metric("MAE", f"{met['MAE']:.3f}")
colB.metric("RMSE", f"{met['RMSE']:.3f}")
colC.metric("MAPE", f"{met['MAPE%']:.2f}%")

# Plot backtest
bt_df = pd.concat([
    train[[value_col]].assign(kind="train"),
    test[[value_col]].assign(kind="test"),
    pd.Series(test_fc, index=test.index, name=value_col).to_frame().assign(kind="test_forecast"),
])
fig_bt = px.line(bt_df.reset_index(), x=df_ts.index.name, y=value_col, color="kind", title="Backtest: train / test / test-forecast")
st.plotly_chart(fig_bt, use_container_width=True)

# -----------------------------
# Forecast next N days
# -----------------------------
st.subheader("4) Forecast next N steps")

future_index = pd.date_range(df_ts.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
future = pd.DataFrame({df_ts.index.name: future_index, "forecast": np.asarray(future_fc)})

fig_fc = px.line(
    pd.concat([
        df_ts.rename(columns={value_col: "value"}).reset_index(),
        future.rename(columns={"forecast": "value"}).assign(kind="forecast"),
    ]).assign(kind=lambda d: d.get("kind", "history")),
    x=df_ts.index.name, y="value", color="kind", title=f"Forecast for next {horizon} steps",
)
st.plotly_chart(fig_fc, use_container_width=True)

st.dataframe(future)

# Download
csv = future.rename(columns={"forecast": f"{value_col}_forecast"}).to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download forecast CSV",
    data=csv,
    file_name="forecast_next_30.csv",
    mime="text/csv",
)

st.caption(
    "Models: Holt‑Winters via statsmodels; optional Prophet if installed. This app resamples to your chosen frequency, "
    "uses a small hold‑out for validation, and outputs a {horizon}-step forecast with interactive plots."
)
