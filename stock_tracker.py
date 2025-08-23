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
import numpy as np
import warnings

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
            print(f"{ticker}: No valid price data")
            return None
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# --- Build DataFrame ---
def build_frame(df):
    if df is None or df.empty:
        return None

    # Ensure Close is a 1D Series
    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # sometimes Close is multi-column
            close = close.iloc[:, 0]  # take first column
    else:
        print("No 'Close' column available")
        return None

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
        print(f"{ticker}: Skipping due to invalid or missing data")
        return

    latest = frame.tail(1).iloc[0]
    current_price = latest["Close"]
    rsi = latest["RSI"]

    if pd.isna(current_price) or pd.isna(rsi):
        print(f"{ticker}: Latest data unavailable")
        return

    # Determine action
    if current_price >= buy_price * (1 + PROFIT_TARGET) or rsi > 70:
        action = "SELL ❌ (profit target reached)"
    elif current_price <= buy_price * (1 - DIP_THRESHOLD) or rsi < 35:
        action = "BUY MORE ✅ (price dipped)"
    else:
        action = "HOLD ➖"

    print(f"{ticker} | Buy Price: €{buy_price:.2f} | Current: €{current_price:.2f} | Shares: {shares} | RSI: {rsi:.2f} | Action: {action}")

# --- Interactive input for multiple stocks ---
personal_stocks = []
print("Enter your stocks (type 'done' when finished):")
while True:
    ticker = input("Ticker: ").strip()
    if ticker.lower() == "done":
        break
    try:
        buy_price = float(input("Buy Price (€): ").strip())
        shares = int(input("Number of Shares: ").strip())
        personal_stocks.append({"ticker": ticker, "buy_price": buy_price, "shares": shares})
    except ValueError:
        print("Invalid input. Try again.")

# --- Run check for all entered stocks ---
print("\n--- Personal Stock Summary ---")
for stock in personal_stocks:
    check_personal_stock(stock["ticker"], stock["buy_price"], stock["shares"])


# In[ ]:




