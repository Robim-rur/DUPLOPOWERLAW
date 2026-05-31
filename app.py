import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="Crypto Quant Engine v3", layout="wide")
st.title("🏦 Crypto Quant Engine v3 — Unified Institutional Model (FIXED)")

# ==========================================================
# SESSION STATE
# ==========================================================
if "signal_log" not in st.session_state:
    st.session_state.signal_log = []

# ==========================================================
# ASSET SELECTION
# ==========================================================
asset = st.sidebar.selectbox("📊 Asset", ["BTC-USD", "BNB-USD"])

st.sidebar.header("🎯 ATR Risk Engine")
gain_atr = st.sidebar.slider("Take Profit (ATR)", 1.0, 10.0, 3.0, 0.5)
loss_atr = st.sidebar.slider("Stop Loss (ATR)", 0.5, 5.0, 1.5, 0.5)

# ==========================================================
# DATA LOADER (FIXED - ROBUST)
# ==========================================================
@st.cache_data(ttl=3600)
def load_data(symbol):
    df = yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    return df


df = load_data(asset)

if df is None or df.empty or len(df) < 200:
    st.error(f"Sem dados suficientes para {asset}. Tente novamente em alguns segundos.")
    st.stop()

# ==========================================================
# INDICATORS CORE
# ==========================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def power_law(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # FIX: usa início real do ativo (não 2009 fixo)
    genesis = df["Date"].min()

    df["Days"] = (df["Date"] - genesis).dt.days.astype(float)
    df = df[df["Days"] > 0].copy()

    x = np.log10(df["Days"].to_numpy())
    y = np.log10(df["Close"].to_numpy())

    slope, intercept = np.polyfit(x, y, 1)
    df["PowerLaw"] = 10 ** (intercept + slope * x)

    return df

# ==========================================================
# BUILD DATASET
# ==========================================================
df = power_law(df)

df["EMA9"] = ema(df["Close"], 9)
df["EMA29"] = ema(df["Close"], 29)
df["EMA69"] = ema(df["Close"], 69)
df["EMA169"] = ema(df["Close"], 169)

df["RSI"] = rsi(df["Close"], 14)
df["ATR"] = atr(df, 14)

df = df.dropna()

# ==========================================================
# STATE VARIABLES
# ==========================================================
price = float(df["Close"].iloc[-1])
ema9 = float(df["EMA9"].iloc[-1])
ema29 = float(df["EMA29"].iloc[-1])
ema69 = float(df["EMA69"].iloc[-1])
ema169 = float(df["EMA169"].iloc[-1])

rsi_now = float(df["RSI"].iloc[-1])
atr_now = float(df["ATR"].iloc[-1])

trend_ok = price > ema169

# ==========================================================
# EMA RIBBON
# ==========================================================
ema_max = max(ema9, ema29, ema69, ema169)
ema_min = min(ema9, ema29, ema69, ema169)

compression = (ema_max - ema_min) / ema69

if ema9 > ema29 > ema69 > ema169:
    ribbon_state = "BULLISH"
elif ema9 < ema29 < ema69 < ema169:
    ribbon_state = "BEARISH"
elif compression < 0.08:
    ribbon_state = "COMPRESSION"
else:
    ribbon_state = "NEUTRAL"

# ==========================================================
# SCORE ENGINE (UNIFIED)
# ==========================================================
trend_score = 60 if trend_ok else 0
momentum_score = np.clip((40 - rsi_now) * 1.5, 0, 25)
quality_score = 15 if rsi_now < 45 else 5 if rsi_now < 55 else 0

if ribbon_state == "BULLISH":
    ribbon_score = 15
elif ribbon_state == "COMPRESSION":
    ribbon_score = 8
elif ribbon_state == "NEUTRAL":
    ribbon_score = 3
else:
    ribbon_score = 0

score = trend_score + momentum_score + quality_score + ribbon_score

# ==========================================================
# STATE MACHINE
# ==========================================================
if not trend_ok:
    state = "BLOCKED"
    signal = f"⛔ BLOQUEADO ({asset} abaixo EMA 169)"

elif score >= 75:
    state = "LONG"
    signal = "🟢 LONG SETUP CONFIRMADO"

elif score >= 50:
    state = "WAIT"
    signal = "🟡 AGUARDAR"

else:
    state = "NO_TRADE"
    signal = "🔴 SEM TRADE"

# ==========================================================
# ATR PROBABILITY ENGINE
# ==========================================================
def probability_engine(df, gain_atr, loss_atr, samples=300):

    wins = 0
    valid = df.iloc[:-100]

    for _ in range(samples):

        idx = np.random.randint(50, len(valid) - 1)

        entry = valid.iloc[idx]
        price = entry["Close"]
        atr_val = entry["ATR"]

        if np.isnan(atr_val) or atr_val == 0:
            continue

        for i in range(1, 60):

            if idx + i >= len(df):
                break

            future = df.iloc[idx + i]["Close"]

            if future >= price + (gain_atr * atr_val):
                wins += 1
                break

            if future <= price - (loss_atr * atr_val):
                break

    return wins / samples if samples > 0 else 0


prob = probability_engine(df, gain_atr, loss_atr)

# ==========================================================
# LOG SYSTEM
# ==========================================================
last = None
if st.session_state.signal_log:
    last_entries = [x for x in st.session_state.signal_log if x["asset"] == asset]
    last = last_entries[-1]["state"] if last_entries else None

entry = {
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "asset": asset,
    "price": price,
    "ema9": ema9,
    "ema29": ema29,
    "ema69": ema69,
    "ema169": ema169,
    "rsi": rsi_now,
    "atr": atr_now,
    "score": score,
    "state": state,
    "signal": signal,
    "ribbon": ribbon_state,
    "prob": prob
}

if last != state:
    st.session_state.signal_log.append(entry)

# ==========================================================
# UI
# ==========================================================
st.subheader(f"📊 {asset}")

if state == "LONG":
    st.success(signal)
elif state == "WAIT":
    st.warning(signal)
else:
    st.error(signal)

st.sidebar.metric("Prob Gain > Loss", f"{prob*100:.1f}%")

# ==========================================================
# METRICS
# ==========================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric(asset, f"${price:,.0f}")
c2.metric("EMA 169", f"${ema169:,.0f}")
c3.metric("Score", f"{score:.1f}/100")
c4.metric("ATR Edge", f"{prob*100:.1f}%")

st.divider()

# ==========================================================
# CHART
# ==========================================================
fig = go.Figure()

fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name=asset))
fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA9"], name="EMA 9"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA29"], name="EMA 29"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA69"], name="EMA 69"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA169"], name="EMA 169"))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["PowerLaw"],
    name="Power Law",
    line=dict(dash="dot")
))

fig.update_layout(height=650, yaxis_type="log")

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# HISTÓRICO
# ==========================================================
st.subheader("📊 Histórico de Sinais")

log_df = pd.DataFrame(st.session_state.signal_log)
log_df = log_df[log_df["asset"] == asset]

if not log_df.empty:
    st.dataframe(log_df, use_container_width=True)

    st.download_button(
        "📥 Baixar histórico",
        log_df.to_csv(index=False),
        file_name=f"{asset}_signal_log.csv",
        mime="text/csv"
    )
else:
    st.info("Sem histórico ainda para este ativo.")

# ==========================================================
# RESUMO
# ==========================================================
st.subheader("Resumo Institucional")

st.write({
    "Asset": asset,
    "Preço": price,
    "Score": score,
    "State": state,
    "Ribbon": ribbon_state,
    "Probabilidade": prob,
    "EMA Alignment": trend_ok
})
