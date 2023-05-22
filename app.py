import pandas as pd
import streamlit as st
from utilities import merge,download


st.title('Google News Financial Analysis')
df=pd.DataFrame()
with st.form("my_form"):
     st.write("Choose the Tick and the duration of the Analysis")
     tick=st.text_input('Company name')
     slider_val = st.slider("duration",1,60)
   

   # Every form must have a submit button.
     submitted = st.form_submit_button("Submit")
     if submitted:
        df=merge(tick,slider_val)
if not df:
   st.dataframe(df)
st.button('Download XLS',on_click=download)
