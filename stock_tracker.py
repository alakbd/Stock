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
#!/usr/bin/env python
# coding: utf-8

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
        # Detect currency from ticker suffix
        df["Currency"] = "USD"
        if ticker.endswith(".IR"):
            df["Currency"] = "EUR"
        return df
    except Exception:
        return None

# --- Build Frame ---
def build_frame(df):
    if df is None or df.empty:
        return None

    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # sometimes multi-column
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    frame = pd.DataFrame(index=df.index)
    frame["Close"] = close
    frame["RSI"] = calculate_RSI(frame["Close"], RSI_PERIOD)
    frame["SMA50"] = frame["Close"].rolling(50, min_periods=1).mean()  # ✅ add this
    return frame

# --- Check personal stock ---
def check_personal_stock(ticker, buy_price, shares):
    df = fetch_data(ticker)
    frame = build_frame(df)
    if frame is None or frame.empty:
        return {"Ticker": ticker, "Error": "No data"}

    latest = frame.tail(1).iloc[0]
    current_price = latest["Close"]
    rsi = latest["RSI"]

    if pd.isna(current_price) or pd.isna(rsi):
        return {"Ticker": ticker, "Error": "No latest data"}

    # --- Safe Profit / Loss ---
    invested = buy_price * shares
    current_value = current_price * shares

    pl_euro = float(current_value - invested) if pd.notna(current_value) else 0.0
    pl_percent = (pl_euro / invested * 100) if invested > 0 else 0.0

    # --- Action ---
    if current_price >= buy_price * (1 + PROFIT_TARGET) or rsi > 70:
        action = "SELL ❌"
    elif current_price <= buy_price * (1 - DIP_THRESHOLD) or rsi < 35:
        action = "BUY MORE ✅"
    else:
        action = "HOLD ➖"

    # --- Currency-aware column ---
    price_label = "Current Price (€)" if ticker.endswith(".IR") else "Current Price ($)"

    return {
        "Ticker": ticker,
        "Buy Price (€)": round(buy_price, 2),
        price_label: round(current_price, 2),
        "Shares": shares,
        "RSI": round(rsi, 2),
        "P/L (€)": round(pl_euro, 2),
        "P/L (%)": round(pl_percent, 2),
        "Action": action,
    }
# --- NEW: Scan a watchlist and suggest buys ---
def analyze_watchlist(tickers, rsi_threshold=35):
    rows = []
    for t in tickers:
        t = t.strip()
        if not t:
            continue
        df = fetch_data(t, period="6mo", interval="1d")
        fr = build_frame(df)
        if fr is None or fr.empty:
            continue

        # compute metrics
        last = fr.tail(1).iloc[0]
        if pd.isna(last["RSI"]) or pd.isna(last["Close"]):
            continue

        # 5-day momentum; protect for short history
        if len(fr) >= 6:
            ret_5d = (fr["Close"].iloc[-1] / fr["Close"].iloc[-6] - 1) * 100
        else:
            ret_5d = np.nan

        # 50d trend slope (last 10 days SMA50 change)
        if fr["SMA50"].notna().sum() >= 10:
            sma50_slope = fr["SMA50"].iloc[-1] - fr["SMA50"].iloc[-10]
        else:
            sma50_slope = np.nan

        rows.append({
            "Ticker": t,
            "Close": round(float(last["Close"]), 2),
            "RSI": round(float(last["RSI"]), 2),
            "5D %": None if pd.isna(ret_5d) else round(float(ret_5d), 2),
            "50D Trend": None if pd.isna(sma50_slope) else round(float(sma50_slope), 2),
        })

    # ✅ Always return two DataFrames
    if not rows:
        empty = pd.DataFrame(columns=["Ticker", "Close", "RSI", "5D %", "50D Trend"])
        return empty, empty

    df_out = pd.DataFrame(rows)

    # Suggestion rule: oversold + momentum stabilizing
    mask_candidate = (df_out["RSI"] <= rsi_threshold) & (df_out["5D %"].fillna(-999) >= 0)
    suggestions = df_out[mask_candidate].copy()

    # Rank by RSI asc, then 5D% desc
    if not suggestions.empty:
        suggestions = suggestions.sort_values(by=["RSI", "5D %"], ascending=[True, False])

    return suggestions, df_out.sort_values(by="RSI")

# --- Streamlit UI ---
st.set_page_config(page_title="📈 Personal Stock Tracker", layout="wide")
st.title("📈 Personal Stock Tracker")
st.markdown("Enter your stocks and check if you should **Buy, Sell, or Hold** based on RSI and profit target rules.")

# --- Stock input form ---
with st.form("stock_form", clear_on_submit=True):
    ticker = st.text_input("Ticker (e.g., AAPL, PTSB.IR)").strip().upper()
    buy_price = st.number_input("Buy Price (€)", min_value=0.0, step=0.01)
    shares = st.number_input("Number of Shares", min_value=1, step=1)
    submitted = st.form_submit_button("Add / Update Stock")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# --- Add or update stock ---
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

# --- Display portfolio ---
if st.session_state.portfolio:
    st.subheader("📊 Your Portfolio")
    results = []
    remove_idx = None  # track which stock to remove

    for idx, stock in enumerate(st.session_state.portfolio):
        res = check_personal_stock(stock["ticker"], stock["buy_price"], stock["shares"])
        res["Last Updated"] = stock.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        results.append(res)

        # Remove button
        if st.button(f"❌ Remove {stock['ticker']}", key=f"remove_{idx}"):
            remove_idx = idx

    # Remove outside the loop
    if remove_idx is not None:
        st.session_state.portfolio.pop(remove_idx)
        st.experimental_rerun()

    # Show table
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

    # Chart for last added ticker
    last_ticker = st.session_state.portfolio[-1]["ticker"]
    df_chart = fetch_data(last_ticker, period="6mo")
    if df_chart is not None:
        st.subheader(f"📉 {last_ticker} Price Chart (6mo)")
        st.line_chart(df_chart["Close"])


 # --- NEW: Suggested Buys (BELOW THE CHART) ---
        st.subheader("🧠 Suggested Buys (RSI & momentum)")
        st.caption("Candidates with **RSI ≤ 35** and **5-day % change ≥ 0** (attempting to catch early reversals; not financial advice).")

        default_watchlist = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, V, MA, NFLX, AMD, INTC, TSM, AVGO, PTSB.IR, AIB.IR, CRH.IR, RDSA.IR"
        watchlist_str = st.text_input(
            "Watchlist (comma-separated tickers):",
            value=default_watchlist,
            help="Add any tickers you want me to scan."
        )
        rsi_cutoff = st.slider("RSI max (oversold threshold)", min_value=20, max_value=50, value=35, step=1)

        if watchlist_str.strip():
            tickers = [t.strip() for t in watchlist_str.split(",")]
            suggestions, ranked_by_rsi = analyze_watchlist(tickers, rsi_threshold=rsi_cutoff)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Suggested Buys**")
                if suggestions is not None and not suggestions.empty:
                    st.dataframe(suggestions.reset_index(drop=True), use_container_width=True)
                else:
                    st.info("No suggestions met the criteria today. Try widening the RSI threshold or expanding your watchlist.")

            with col2:
                st.markdown("**All – ranked by RSI (lowest first)**")
                if ranked_by_rsi is not None and not ranked_by_rsi.empty:
                    st.dataframe(ranked_by_rsi.reset_index(drop=True), use_container_width=True)
                else:
                    st.write("—")

else:
    st.info("Add at least one stock to see your portfolio and chart.")

# In[ ]:




