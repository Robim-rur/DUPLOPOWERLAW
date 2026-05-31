import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="Crypto Quant Desk v4 FIXED", layout="wide")
st.title("🏦 Crypto Quant Desk v4 — Multi-Asset Institutional Scanner (FIXED)")

# ==========================================================
# SESSION STATE
# ==========================================================
if "signal_log" not in st.session_state:
    st.session_state.signal_log = []

# ==========================================================
# UNIVERSE
# ==========================================================
ASSETS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "LINK-USD", "XRP-USD"]

# ==========================================================
# RISK ENGINE
# ==========================================================
st.sidebar.header("🎯 ATR Risk Engine")
gain_atr = st.sidebar.slider("Take Profit (ATR)", 1.0, 10.0, 3.0, 0.5)
loss_atr = st.sidebar.slider("Stop Loss (ATR)", 0.5, 5.0, 1.5, 0.5)

# ==========================================================
# DATA LOADER
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

    return df.reset_index()

# ==========================================================
# INDICATORS
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

    genesis = df["Date"].min()
    df["Days"] = (df["Date"] - genesis).dt.days.astype(float)

    df = df[df["Days"] > 0].copy()

    x = np.log10(df["Days"].to_numpy())
    y = np.log10(df["Close"].to_numpy())

    slope, intercept = np.polyfit(x, y, 1)
    df["PowerLaw"] = 10 ** (intercept + slope * x)

    return df

# ==========================================================
# ENGINE PER ASSET
# ==========================================================
def analyze_asset(asset):

    df = load_data(asset)
    if df is None or df.empty or len(df) < 200:
        return None

    df = power_law(df)

    df["EMA9"] = ema(df["Close"], 9)
    df["EMA29"] = ema(df["Close"], 29)
    df["EMA69"] = ema(df["Close"], 69)
    df["EMA169"] = ema(df["Close"], 169)

    df["RSI"] = rsi(df["Close"], 14)
    df["ATR"] = atr(df, 14)

    df = df.dropna()

    price = float(df["Close"].iloc[-1])
    ema9 = float(df["EMA9"].iloc[-1])
    ema29 = float(df["EMA29"].iloc[-1])
    ema69 = float(df["EMA69"].iloc[-1])
    ema169 = float(df["EMA169"].iloc[-1])
    rsi_now = float(df["RSI"].iloc[-1])

    trend_ok = price > ema169

    ema_max = max(ema9, ema29, ema69, ema169)
    ema_min = min(ema9, ema29, ema69, ema169)
    compression = (ema_max - ema_min) / ema69

    if ema9 > ema29 > ema69 > ema169:
        ribbon = "BULLISH"
    elif ema9 < ema29 < ema69 < ema169:
        ribbon = "BEARISH"
    elif compression < 0.08:
        ribbon = "COMPRESSION"
    else:
        ribbon = "NEUTRAL"

    trend_score = 60 if trend_ok else 0
    momentum_score = np.clip((40 - rsi_now) * 1.5, 0, 25)
    quality_score = 15 if rsi_now < 45 else 5 if rsi_now < 55 else 0
    ribbon_score = 15 if ribbon == "BULLISH" else 8 if ribbon == "COMPRESSION" else 3 if ribbon == "NEUTRAL" else 0

    score = trend_score + momentum_score + quality_score + ribbon_score

    def probability_engine(df, gain_atr, loss_atr, samples=250):

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

    final_score = (score * 0.7) + (prob * 100 * 0.3)

    return {
        "asset": asset,
        "df": df,
        "price": price,
        "score": score,
        "prob": prob,
        "final_score": final_score,
        "ribbon": ribbon,
        "trend": trend_ok,
        "rsi": rsi_now
    }

# ==========================================================
# RUN ALL ASSETS
# ==========================================================
results = []

for asset in ASSETS:
    res = analyze_asset(asset)
    if res:
        results.append(res)

df_results = pd.DataFrame(results)
df_ranked = df_results.sort_values("final_score", ascending=False)

# ==========================================================
# UI RANKING
# ==========================================================
st.subheader("🥇 Ranking de Oportunidades (Score + Probabilidade)")

st.dataframe(df_ranked[["asset", "price", "score", "prob", "final_score", "ribbon"]])

top = df_ranked.iloc[0]

st.success(f"🔥 TOP SETUP: {top['asset']} | Score: {top['score']:.1f} | Prob: {top['prob']*100:.1f}%")

st.divider()

# ==========================================================
# CHART (FIXED — EMAs RESTAURADAS)
# ==========================================================
st.subheader(f"📊 Detalhe Completo: {top['asset']}")

df_top = top["df"]

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_top["Date"], y=df_top["Close"], name="Price"))

fig.add_trace(go.Scatter(x=df_top["Date"], y=df_top["EMA9"], name="EMA 9"))
fig.add_trace(go.Scatter(x=df_top["Date"], y=df_top["EMA29"], name="EMA 29"))
fig.add_trace(go.Scatter(x=df_top["Date"], y=df_top["EMA69"], name="EMA 69"))
fig.add_trace(go.Scatter(x=df_top["Date"], y=df_top["EMA169"], name="EMA 169"))

fig.add_trace(go.Scatter(
    x=df_top["Date"],
    y=df_top["PowerLaw"],
    name="Power Law",
    line=dict(dash="dot")
))

fig.update_layout(height=650, yaxis_type="log")

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# SUMMARY
# ==========================================================
st.subheader("Resumo do Desk")

st.write({
    "Top Asset": top["asset"],
    "Final Score": top["final_score"],
    "Probabilidade ATR": top["prob"],
    "Modelo": "EMA Ribbon + RSI + ATR Probability + Power Law"
})
