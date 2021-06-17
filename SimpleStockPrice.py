import yfinance as yf
import pandas as pd
import streamlit as sl

sl.write("A simple stock price app")

tickerSymbol = 'GOOG'

tickerData = yf.Ticker(tickerSymbol)
tickerdf = tickerData.history(period='', start='2020-05-27', end='2020-12-31')

sl.line_chart(tickerdf.Close)
sl.line_chart(tickerdf.Volume)
