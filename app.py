import pandas as pd
import streamlit as st
from utilities import fetch,download


st.title('News Financial Analysis')
df=pd.DataFrame()
with st.form("my_form"):
     st.write("Choose the Tick and the duration of the Analysis")
     tick=st.text_input('Company name')
     slider_val = st.slider("duration",1,6)
   

   # Every form must have a submit button.
     submitted = st.form_submit_button("Submit")
     if submitted:
        df=fetch(tick,slider_val)
        print(df.columns)
        df=df.reindex(columns=['title','source','Pos','Neg','Neutral','Open','Close','Volume','High','Low','Adj Close','desc','article'])
        df.reset_index(inplace=True)
        
        st.dataframe(df.style.highlight_max(axis=1,subset=['Pos','Neg','Neutral'], props='color:white; font-weight:bold; background-color:darkblue;'))
#st.dataframe(df)
if not df.empty :
  
        st.download_button(
          label="Download XLS",
          data=download(),
          file_name="NewsFinancialAnalysis.xlsx",
          mime="application/vnd.ms-excel"
        )

#st.button('Download XLS',on_click=download)

