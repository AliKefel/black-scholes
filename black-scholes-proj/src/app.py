import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt





st.set_page_config(layout="wide")

st.sidebar.title("PARAMETERS📊")

st.sidebar.markdown("<div style='text-align: start; margin-top: 20px;  font-size:large;'> Created by Ali Kefel </div>", unsafe_allow_html=True)



def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return  put_price


    
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

st.sidebar.header("Input Parameters")
S = float(st.sidebar.text_input("Stock Price (S)", "100"))
K = float(st.sidebar.text_input("Strike Price (K)", "100"))
T = float(st.sidebar.text_input("Time to Maturity (T) in years", "1"))
r = float(st.sidebar.text_input("Risk-Free Rate (r)", "0.05"))
sigma = float(st.sidebar.text_input("Volatility (σ)", "0.2"))

st.sidebar.markdown("---")

min_spot_price = st.sidebar.slider("Minimum Spot Price (S)", min_value=0, max_value=1000, value=50)
max_spot_price = st.sidebar.slider("Maximum Spot Price (S)", min_value=0, max_value=1000, value=150)
spot_price_step = st.sidebar.slider("Spot Price Step", min_value=1, max_value=100, value=10)

min_volatility = st.sidebar.slider("Minimum Volatility (σ)", min_value=0.0, max_value=1.0, value=0.1)
max_volatility = st.sidebar.slider("Maximum Volatility (σ)", min_value=0.0, max_value=1.0, value=0.5)
volatility_step = st.sidebar.slider("Volatility Step", min_value=0.01, max_value=0.1, value=0.05)


# Example of a button
calculate = st.sidebar.button("Calculate")

st.header("Results")

# Create a DataFrame to display the parameters and their values
data = {
    "Stock Price (S)": [S],
    "Strike Price (K)": [K],
    "Time to Maturity (T)": [T],
    "Risk-Free Rate (r)": [r],
    "Volatility (σ)": [sigma]
}

df = pd.DataFrame(data)

st.table(df)


# Perform the calculation and display results only if the button is clicked
if calculate:
    call_price, put_price = black_scholes(S, K, T, r, sigma)
    
    # Display the call and put prices as large colored buttons
    st.markdown(f"""
        <style>
        .price-button {{
            font-size: 24px;
            width: 100%;
            height: 100px;
            margin-top: 20px;
            border: none;
            color: white;
            text-align: center;
            line-height: 100px;
            border-radius: 10px;
        }}
        .call-button {{
            background-color: green;
        }}
        .put-button {{
            background-color: red;
        }}
        </style>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="price-button call-button">Call Option Price: {call_price:.2f}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="price-button put-button">Put Option Price: {put_price:.2f}</div>', unsafe_allow_html=True)

spot_prices = np.arange(min_spot_price, max_spot_price + spot_price_step, spot_price_step)
volatilities = np.arange(min_volatility, max_volatility + volatility_step, volatility_step)

# Create DataFrames to store call and put option prices
call_prices = pd.DataFrame(index=volatilities, columns=spot_prices)
put_prices = pd.DataFrame(index=volatilities, columns=spot_prices)

# Calculate call and put option prices for each combination of spot price and volatility
for sigma in volatilities:
    for S in spot_prices:
        call_prices.at[sigma, S] = black_scholes_call(S, K, T, r, sigma)
        put_prices.at[sigma, S] = black_scholes_put(S, K, T, r, sigma)

call_prices = call_prices.astype(float)
put_prices = put_prices.astype(float)

# Main content area for displaying the heatmap
st.header("Black-Scholes Option Price Heatmaps: ")

# Display the heatmaps side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Option Price Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f")
    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Volatility (σ)")
    ax.set_yticklabels([f"{y:.2f}" for y in call_prices.index])
    st.pyplot(fig)

with col2:
    st.subheader("Put Option Price Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f")
    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Volatility (σ)")
    ax.set_yticklabels([f"{y:.2f}" for y in put_prices.index])
    st.pyplot(fig)
    
