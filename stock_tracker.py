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

import yfinance as yf
import pandas as pd
import streamlit as st

# --- Functions ---
def fetch_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d")
        if df.empty:
            return None
        return df
    except:
        return None

def build_frame(df):
    frame = pd.DataFrame(index=df.index)
    frame["Close"] = df["Close"]
    frame["MA20"] = df["Close"].rolling(window=20).mean()
    frame["MA50"] = df["Close"].rolling(window=50).mean()
    return frame

def generate_signal(frame, buy_price=None):
    if frame is None or frame.empty:
        return "No Data", None
    latest = frame.iloc[-1]
    price = latest["Close"]

    # Simple rule: MA crossover
    if latest["MA20"] > latest["MA50"]:
        signal = "BUY ✅"
    elif latest["MA20"] < latest["MA50"]:
        signal = "SELL ❌"
    else:
        signal = "HOLD ➖"

    # Profit check if buy_price is given
    profit = None
    if buy_price:
        profit = (price - buy_price) / buy_price * 100

    return signal, profit

# --- Streamlit UI ---
st.title("📈 Personal Stock Tracker")

# User inputs
ticker = st.text_input("Enter Stock Ticker (e.g. PTSB.IR, AAPL):")
buy_price = st.number_input("Enter your Buy Price (€):", min_value=0.0, format="%.2f")
shares = st.number_input("Number of Shares:", min_value=0, step=1)

if st.button("Check Stock"):
    df = fetch_data(ticker)
    if df is None:
        st.error("No data found for this ticker.")
    else:
        frame = build_frame(df)
        signal, profit = generate_signal(frame, buy_price)

        latest_price = frame["Close"].iloc[-1]

        st.subheader(f"📊 {ticker} Analysis")
        st.write(f"**Latest Price:** €{latest_price:.2f}")
        st.write(f"**Signal:** {signal}")

        if buy_price > 0:
            st.write(f"**Your Buy Price:** €{buy_price:.2f}")
            st.write(f"**Shares Held:** {shares}")
            st.write(f"**Profit/Loss:** {profit:.2f}%")


# In[ ]:




