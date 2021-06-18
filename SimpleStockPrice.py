import pandas as pd
import streamlit as sl
import pandas_datareader .data as web

sl.write("A simple stock price app")

sdate = dt.datetime(2020,1,1)
edate = dt.datetime.now()
absa = web.DataReader('ABG.JO', data_source='yahoo', start=sdate, end=edate)

sl.line_chart(absa.Close)
sl.line_chart(absa.Volume)
