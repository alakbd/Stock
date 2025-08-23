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

# --- RSI Calculation ---
def calculate_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Data Fetch ---
def fetch_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d")
        if df.empty:
            return None
        return df
    except:
        return None

# --- Build Frame ---
def build_frame(df, ma_period=50, rsi_period=14):
    frame = pd.DataFrame(index=df.index)
    frame["Close"] = df["Close"]
    frame[f"MA{ma_period}"] = df["Close"].rolling(window=ma_period).mean()
    frame["RSI"] = calculate_RSI(df["Close"], rsi_period)
    return frame

# --- Signal Generation ---
def generate_signal(row, ma_period=50):
    if row["Close"] > row[f"MA{ma_period}"] and row["RSI"] < 30:
        return "BUY ✅"
    elif row["Close"] < row[f"MA{ma_period}"] and row["RSI"] > 70:
        return "SELL ❌"
    else:
        return "HOLD ➖"

# --- Streamlit UI ---
st.title("📈 Personal Stock Tracker (MA50 + RSI Strategy)")

ticker = st.text_input("Enter Stock Ticker (e.g. PTSB.IR, AAPL):")
buy_price = st.number_input("Enter your Buy Price (€):", min_value=0.0, format="%.2f")
shares = st.number_input("Number of Shares:", min_value=0, step=1)

if st.button("Check Stock"):
    df = fetch_data(ticker, period="1y")
    if df is None:
        st.error("No data found for this ticker.")
    else:
        frame = build_frame(df)
        latest = frame.iloc[-1]
        signal = generate_signal(latest)

        st.subheader(f"📊 {ticker} Analysis")
        st.write(f"**Latest Price:** €{latest['Close']:.2f}")
        st.write(f"**MA50:** €{latest['MA50']:.2f}")
        st.write(f"**RSI:** {latest['RSI']:.2f}")
        st.write(f"**Signal:** {signal}")

        if buy_price > 0:
            profit = (latest['Close'] - buy_price) / buy_price * 100
            st.write(f"**Your Buy Price:** €{buy_price:.2f}")
            st.write(f"**Shares Held:** {shares}")
            st.write(f"**Profit/Loss:** {profit:.2f}%")



# In[ ]:




