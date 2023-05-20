import streamlit as st
import yfinance as yf
#from date import uu
from utilities import get_news_data,clean_date,SentimentAnalyzer,get_stock_data,merge

from pandas.tseries.offsets import BDay
import numpy as np
import datetime

st.title('Google News Financial Analysis')
with st.form("my_form"):
   st.write("Choose the Tick and the duration of the Analysis")
   tick=st.text_input('Company name')
   slider_val = st.slider("duration",1,60)
   

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
      df=merge(tick,slider_val)
st.dataframe(df)

