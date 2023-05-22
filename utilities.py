import streamlit as st
import re
import numpy as np
import pandas as pd
import yfinance as yf
from newspaper import Config
from tqdm import tqdm
from GoogleNews import GoogleNews
#import datetime
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
            'pr': 'Apr'
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
  # Define the columns you want in your DataFrame
  columns = ["title", "datetime", "desc", "source", "article", "keywords", "Pos", "Neg", "Neutral"]

  # Create an empty DataFrame with the columns you defined
  df = pd.DataFrame(columns=columns)

  # Create a GoogleNews object
  googlenews = GoogleNews()

  # Create an empty set to store unique article titles
  unique_titles = set()

  # Loop through each company and get the news articles
  
  for i in tqdm(range(1, int(days/10))):
      googlenews.search(company + " financial news")
      googlenews.getpage(i)
      results = googlenews.result()
      
      #new df
      re_df = pd.DataFrame(results)
      orig=re_df
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


def merge(company,days):
  news_data=get_news_data(company,days)
  stock_data=get_stock_data(company)
  news_data.index = pd.to_datetime(news_data.index)
  stock_data.index=pd.to_datetime(stock_data.index)
  asof=merge_asof(news_data.sort_values('Date'),stock_data.sort_values('Date'),on='Date',allow_exact_matches=False)
  # convert the 'Date' column to datetime format
  asof['Date'] = pd.to_datetime(asof['Date'])

  # set the 'Date' column as the index
  asof.set_index('Date', inplace=True)

  # sort the index in ascending order
  # sort the index in descending order
  asof.sort_index(ascending=False, inplace=True)
  nd=news_data
  sd=stock_data
  asod=asof
  return asof
def download():
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    writer = pd.ExcelWriter(desktop + '/output.xlsx', engine='xlsxwriter')
    #writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    nd.to_excel(writer, sheet_name='news_data')
    sd.to_excel(writer, sheet_name='stock_data')
    asod.to_excel(writer, sheet_name='merge_data')
    orig.to_excel(writer,sheet_name='results')

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()