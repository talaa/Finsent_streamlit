import os
import streamlit as st
import re
import numpy as np
import pandas as pd
import yfinance as yf
from newspaper import Config
from stqdm import stqdm
from GoogleNews import GoogleNews
from io import BytesIO
from datetime import datetime
from newspaper import Article
from pandas.core.reshape.merge import merge_asof
from pandas import ExcelWriter
#### Sentiment Requirements 
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import torch.nn.functional as F


user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config = Config()

config.browser_user_agent = user_agent
import nltk
nltk.download('punkt')
nd=pd.DataFrame()
sd=pd.DataFrame()
asod=pd.DataFrame()
orig=pd.DataFrame()

def clean_date(date_string):
    try:
        # Try parsing date as '9-Apr-2023'
        return datetime.strptime(date_string, '%d-%b-%Y').date()
        
    except ValueError:
        # Try parsing date using regular expressions and handle missing/abbreviated month names
        month_mapping = {
            'ar': 'Mar',
            'un': 'Jun',
            'an': 'Jan',
            'ct': 'Oct',
            'pr': 'Apr',
            'ul': 'Jul',
            'eb': 'Feb',
            'ay': 'May',
            'ug': 'Aug',
            'ep': 'Sep',
            'ov': 'Nov',
            'ec': 'Dec'
            # Add more mappings as needed for missing or abbreviated month names
        }
        #match = re.search(r'(\D+)(\d+)(?:,\s)?(\d+)', date_string)
        match = re.search(r'\s*(\D+)\s+(\d+)(?:,\s)?(\d+)', date_string)
        if match:
            month = match.group(1)
            if month.lower() in month_mapping:
                month = month_mapping[month.lower()]
            day = match.group(2)
            year = match.group(3)
            return datetime.strptime(f'{month} {day}, {year}', '%b %d, %Y').date()
            
        else:
            raise ValueError("Invalid date format")
@st.cache_data        
def get_news_data(company,days):
    #setting variables 
    pages=5
    # get the company Name 
    co = yf.Ticker(company)
    company_name = co.info['longName']
    # Define the columns you want in your DataFrame
    columns = ["title", "datetime", "desc", "source", "article", "keywords", "Pos", "Neg", "Neutral"]

    # Create an empty DataFrame with the columns you defined
    df = pd.DataFrame(columns=columns)

    # Create a GoogleNews object
    googlenews = GoogleNews()
    period=str(days)+'d'
    googlenews.setperiod(period) 
    googlenews.search(company_name + " financial news")
    num_pages = googlenews.total_count()
    
    results = googlenews.result()
    
    #re_df.drop_duplicates(subset=['title'], inplace=True)
    print('the len of results is ',len(results),' The num of pages is ',num_pages)

    # Create an empty set to store unique article titles
    unique_titles = set()
    st.info('we are analyzing multiple articles,  please wait !',icon="ℹ️")
    # Create an empty list to store the news articles
    news_articles = []
    # Create an empty list to store the news articles, icon="ℹ️")

    # Loop through each company and get the news articles
    for i in stqdm(range(1, pages)):
    #for i in tqdm(range(1, int(days/10))):
    
        # Set the period and language of the news articles you want to get
        
        googlenews.getpage(i)
        
        
        #new df
        re_df = pd.DataFrame(results)
        re_df['Tick']=company
        st.session_state.orig=re_df
        re_df.replace('', np.nan, inplace=True)
        re_df.dropna(subset=['title'], inplace=True)
        # Convert DateTimeColumn to datetime
        re_df['datetime'] = pd.to_datetime(re_df['datetime'])

        # Create a new column with cleaned dates
        re_df['cleaned_date'] = re_df['datetime'].dt.date


        # Clean and transform dates in DateColumn only if corresponding entry in DateTimeColumn is empty
        re_df.loc[re_df['datetime'].isna(), 'cleaned_date'] = re_df.loc[re_df['datetime'].isna(), 'date'].apply(clean_date)
        re_df.drop_duplicates(subset=['title'], inplace=True)
        re_df.drop(['date','datetime','img'], axis=1, inplace=True)
        re_df.rename(columns={'cleaned_date':'datetime'}, inplace=True)
        
        

        # Loop through each article and add it to the DataFrame
        for index, row in re_df.iterrows():
            result = row.to_dict()
            date_str = result["datetime"].strftime('%Y-%m-%d %H:%M:%S.%f')
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
            if (datetime.now() - date_obj).days <= days :
                
                article = Article(result["link"],config=config)
                try:
                    article.download()
                    article.parse()
                except Exception as e:
                    print(f"Error while downloading article: {e}")
                    continue
                unique_titles.add(result['title'])
                
                article.nlp()
                sentiment = SentimentAnalyzer(article.text)
                #print(result['title'])
                df = pd.concat([df, pd.DataFrame({
                    "title": [result["title"]],
                    "datetime": [result["datetime"]],
                    "desc": [result["desc"]],
                    "source": [result["media"]],
                    "article": [result["link"]],
                    "keywords": [', '.join(article.keywords)],
                    "Pos": [sentiment[0][0]],
                    "Neg": [sentiment[0][1]],
                    "Neutral": [sentiment[0][2]]
                })])

    # Print the DataFrame
    #df['datetime'] = pd.to_datetime(df['datetime'],unit='s')
    df[['Pos', 'Neg', 'Neutral']] = df[['Pos', 'Neg', 'Neutral']].apply(lambda x: x*100)
    df.drop_duplicates(subset=['title'], inplace=True)
    #ss_df = df.reset_index()
    #df.drop('index', axis=1, inplace=True)
    df.set_index('datetime', inplace=True)
    df.index.rename('Date', inplace=True)
    
    return df
@st.cache_data
def SentimentAnalyzer(doc):
    pt_batch = tokenizer(doc,padding=True,truncation=True,max_length=512,return_tensors="pt")
    outputs = model(**pt_batch)
    pt_predictions = F.softmax(outputs.logits, dim=-1)
    return pt_predictions.detach().cpu().numpy()
@st.cache_data
def get_stock_data(Tick):
    #today = datetime.date.today()
    today = datetime.today().date()
    tick = yf.download(Tick, '2023-1-1', today)
    return tick

@st.cache_data
def fetch(company,days):
    news_data=get_news_data(company,days)
    stock_data=get_stock_data(company)
    news_data.index = pd.to_datetime(news_data.index)
    stock_data.index=pd.to_datetime(stock_data.index)
    asof=merge_asof(news_data.sort_values('Date'),stock_data.sort_values('Date'),on='Date',allow_exact_matches=False)
    # convert the 'Date' column to datetime format
    asof['Date'] = pd.to_datetime(asof['Date'])

    # set the 'Date' column as the index
    asof.set_index('Date', inplace=True)
    co = yf.Ticker(company)
    company_name = co.info['longName']
    #st.write('we Have analyzed '+str(len(asof))+' articles about'+company_name+'from Today till '+days+'d ago')
    # adding the name of the tick 
    news_data['Tick']=company
    stock_data['Tick']=company
    asof['Tick']=company
    # sshow the results 
    st.success('we Have analyzed '+str(len(asof))+' articles about '+company_name+'from Today till '+str(days)+'d ago '+'from '+str(asof['source'].nunique())+' different sources',icon="✅")
    # sort the index in descending order
    asof.sort_index(ascending=False, inplace=True)
    st.session_state.nd=news_data
    st.session_state.sd=stock_data
    st.session_state.asod=asof
    return asof

def download():
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    #desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    #writer = pd.ExcelWriter(desktop + '/output.xlsx', engine='xlsxwriter')
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    st.session_state.nd.to_excel(writer, sheet_name='news_data')
    st.session_state.sd.to_excel(writer, sheet_name='stock_data')
    st.session_state.asod.to_excel(writer, sheet_name='merge_data')
    st.session_state.orig.to_excel(writer,sheet_name='results')

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()
    processed_data = output.getvalue()
    return processed_data
