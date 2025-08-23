#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Example: Positive earnings news → Buy signal.
#Sudden lawsuit/CEO resignation → Sell signal.

 ##Close Price vs Moving Averages

#If 20-day Moving Average (MA20) crosses above 50-day Moving Average (MA50) → Buy Signal (uptrend).

#If MA20 falls below MA50 → Sell Signal (downtrend).

##RSI (Relative Strength Index)

#RSI > 70 → Overbought → Sell Signal

#RSI < 30 → Oversold → Buy Signal

#RSI between 30–70 → Neutral / Hold

# stock_tracker_app.py
#!/usr/bin/env python
# coding: utf-8

# stock_tracker_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Parameters ---
RSI_PERIOD = 14
PROFIT_TARGET = 0.05  # 5% gain to sell
DIP_THRESHOLD = 0.03  # 3% drop to buy more

# --- RSI Calculation ---
def calculate_RSI(series, period=14):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < period:
        return pd.Series([np.nan] * len(series), index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Fetch Data ---
def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None

# --- Build Frame ---
def build_frame(df):
    if df is None or df.empty:
        return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    frame = pd.DataFrame(index=df.index)
    frame["Close"] = close
    frame["RSI"] = calculate_RSI(frame["Close"], RSI_PERIOD)
    return frame

# --- Check personal stock ---
def check_personal_stock(ticker, buy_price, shares):
    df = fetch_data(ticker)
    frame = build_frame(df)
    if frame is None:
        return {"Ticker": ticker, "Error": "No data"}

    latest = frame.tail(1).iloc[0]
    current_price = latest["Close"]
    rsi = latest["RSI"]

    if pd.isna(current_price) or pd.isna(rsi):
        return {"Ticker": ticker, "Error": "No latest data"}

    # Profit / loss
    pl_euro = (current_price - buy_price) * shares
    pl_percent = ((current_price - buy_price) / buy_price) * 100

    # Determine action
    if current_price >= buy_price * (1 + PROFIT_TARGET) or rsi > 70:
        action = "SELL ❌"
    elif current_price <= buy_price * (1 - DIP_THRESHOLD) or rsi < 35:
        action = "BUY MORE ✅"
    else:
        action = "HOLD ➖"

    return {
        "Ticker": ticker,
        "Buy Price (€)": round(buy_price, 2),
        "Current Price (€)": round(current_price, 2),
        "Shares": shares,
        "RSI": round(rsi, 2),
        "P/L (€)": round(pl_euro, 2),
        "P/L (%)": round(pl_percent, 2),
        "Action": action,
    }

# --- Streamlit UI ---
st.set_page_config(page_title="📈 Personal Stock Tracker", layout="wide")
st.title("📈 Personal Stock Tracker")

st.markdown("Enter your stocks and check if you should **Buy, Sell, or Hold** based on RSI and profit target rules.")

with st.form("stock_form", clear_on_submit=True):
    ticker = st.text_input("Ticker (e.g., AAPL, PTSB.IR)").strip().upper()
    buy_price = st.number_input("Buy Price (€)", min_value=0.0, step=0.01)
    shares = st.number_input("Number of Shares", min_value=1, step=1)
    submitted = st.form_submit_button("Add / Update Stock")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if submitted and ticker and buy_price > 0 and shares > 0:
    found = False
    for stock in st.session_state.portfolio:
        if stock["ticker"] == ticker:
            stock["buy_price"] = buy_price
            stock["shares"] = shares
            stock["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            found = True
            break
    if not found:
        st.session_state.portfolio.append(
            {
                "ticker": ticker,
                "buy_price": buy_price,
                "shares": shares,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

if st.session_state.portfolio:
    st.subheader("📊 Your Portfolio")
    results = []
    for stock in st.session_state.portfolio:
        res = check_personal_stock(stock["ticker"], stock["buy_price"], stock["shares"])
        res["Last Updated"] = stock["timestamp"]
        results.append(res)

    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

    # Optional: show stock chart for the last selected ticker
    last_ticker = st.session_state.portfolio[-1]["ticker"]
    df_chart = fetch_data(last_ticker, period="6mo")
    if df_chart is not None:
        st.subheader(f"📉 {last_ticker} Price Chart (6mo)")
        st.line_chart(df_chart["Close"])

# In[ ]:




