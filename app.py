from array import array
from ast import Return
from cProfile import label
import os
import math
import warnings
from matplotlib import markers
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, json
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import statsmodels.api as sm
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model

# Sentiment analysis libraries
import re
import tweepy 
import nltk
import pandas as pd
import matplotlib.pyplot as plt
# from wordcloud import WordCloud,STOPWORDS
nltk.download('punkt')   
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tweepy import OAuthHandler 
from textblob import TextBlob
from yahoo_fin.stock_info import *


app=Flask(__name__)
# model=None

@app.before_first_request
def before_first_request_func():
    global model
    model = load_model('weight/model.h5')
    print("Model Loaded")

@app.route('/')
def home():
    bitcoin = get_quote_data('btc-usd')['regularMarketPrice']
    ethereum = get_quote_data('eth-usd')['regularMarketPrice']
    matic = get_quote_data('matic-usd')['regularMarketPrice']
    return render_template('index.html', bitcoin=bitcoin, ethereum=ethereum, matic=matic)

@app.route('/plot-ml')
def plot_ml_png():
    v1 = request.args.get('v1', type=float)
    v2 = request.args.get('v2', type=float)
    v3 = request.args.get('v3', type=float)
    v4 = request.args.get('v4', type=float)
    v5 = request.args.get('v5', type=float)
    five_days_pred = [v1,v2,v3,v4,v5]
    fig = create_ml_plot(five_days_pred)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_ml_plot(five_days_pred):
    fig, ax = plt.subplots(figsize=(16,7))
    ax.plot(five_days_pred, 'r', marker='.', label='Prediction for next 5 days')
    plt.tight_layout()
    plt.legend()
    return fig

    
@app.route('/plot-sentiment')
def plot_sentiment_png():
    positive = request.args.get('positive', type=float)
    negative = request.args.get('negative', type=float)
    split = np.array([positive,negative])
    fig = create_pie_chart(split)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_pie_chart(split):
    mylabels = ["Positive", "Negative"]
    myexplode = [0.2, 0]
    fig, ax = plt.subplots(figsize=(16,7))
    ax.pie(split, labels = mylabels, explode=myexplode)
    plt.tight_layout()
    plt.legend(loc="lower left", bbox_to_anchor=(0,0,1,1))
    return fig

@app.route('/prediction',methods=['POST'])
def prediction():
    coin = request.form["coin"]
    if(coin == "BTC-USD"):
        frontend_query = "bitcoin"
    elif(coin == "MATIC-USD"):
        frontend_query = "matic" # Decide the query param
    else:
        frontend_query = "ethereum"

    end_date = date.today()
    start_date = end_date - timedelta(days=180)
    btc_input_df = yf.download(coin,interval='1h',start=start_date, end=end_date)
    btc_input_df.reset_index(level=0, inplace=True)
    btc_input_df.rename(columns = {'index':'time'}, inplace = True)
    btc_input_df_datetype = btc_input_df
    btc_input_df_datetype.rename(columns = {'Close':'close'}, inplace = True)
    
    #Mean of prices
    btc_input_df_datetype['date'] = pd.to_datetime(btc_input_df_datetype['time'],unit='s').dt.date
    group = btc_input_df_datetype.groupby('date')
    btc_closing_price_groupby_date = group['close'].mean()

    prediction_days = 30
    # Set Train data to be uplo ( Total data length - prediction_days )
    df_train= btc_closing_price_groupby_date[:len(btc_closing_price_groupby_date)-prediction_days].values.reshape(-1,1)
    # Set Test data to be the last prediction_days (or 60 days in this case)
    df_test= btc_closing_price_groupby_date[len(btc_closing_price_groupby_date)-prediction_days:].values.reshape(-1,1)

    #Normalizing
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler_train.fit_transform(df_train)
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    scaled_test = scaler_test.fit_transform(df_test)

    #TimeSeries
    trainX, trainY = dataset_generator_lstm(scaled_train)
    testX, testY = dataset_generator_lstm(scaled_test)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1 ))

    #Predict

    predicted_btc_price_test_data = model.predict(testX)
    predicted_btc_price_test_data = scaler_test.inverse_transform(predicted_btc_price_test_data.reshape(-1, 1))
    test_actual = scaler_test.inverse_transform(testY.reshape(-1, 1))

    
    lookback_period = 5
    testX_last_5_days = testX[testX.shape[0] - lookback_period :  ]
    predicted_5_days_forecast_price_test_x = []

    for i in range(5):  
        predicted_forecast_price_test_x = model.predict(testX_last_5_days[i:i+1])
        
        predicted_forecast_price_test_x = scaler_test.inverse_transform(predicted_forecast_price_test_x.reshape(-1, 1))
        # print(predicted_forecast_price_test_x)
        predicted_5_days_forecast_price_test_x.append(predicted_forecast_price_test_x)

    # Converting into a numpy array
    predicted_5_days_forecast_price_test_x = np.array(predicted_5_days_forecast_price_test_x)
    predicted_5_days_forecast_price_test_x = predicted_5_days_forecast_price_test_x.flatten()

    #Plotting prediction days

    mean_predicted =predicted_5_days_forecast_price_test_x.mean()

    #Sentimental Analysis 

    tweets = fetch_tweets(query = frontend_query, count = 100)
    # Converting the list into a pandas Dataframe
    df_sentiment = pd.DataFrame(tweets, columns= ['tweets', 'clean_tweets','sentiment'])

    # Dropping the duplicate values just in case there are some tweets that are copied and then stores the data in a csv file
    df_sentiment = df_sentiment.drop_duplicates(subset='clean_tweets')
    df_sentiment.to_csv('data.csv', index= False)
    ptweets = df_sentiment[df_sentiment['sentiment'] == 'positive']
    p_perc = 100 * len(ptweets)/len(tweets)
    ntweets = df_sentiment[df_sentiment['sentiment'] == 'negative']
    n_perc = 100 * len(ntweets)/len(tweets)
    neutral_perc = 100 - p_perc - n_perc
    p_perc = p_perc + neutral_perc/2
    n_perc = n_perc + neutral_perc/2
    

    # Calculation of both the models
    returns = (predicted_5_days_forecast_price_test_x.mean()/get_quote_data(coin)['regularMarketPrice'])-1
    weight_ml = 80
    weight_crypto = 100 - weight_ml

    # calculating recommendation score
    recommendation_score = (returns*weight_ml + p_perc/100*weight_crypto)

    if(predicted_5_days_forecast_price_test_x.mean() > get_quote_data('btc-usd')['regularMarketPrice'] and p_perc > n_perc):
        final_result= "<strong>Most probably if you buy this crypto currency you will make profit. Algorithm is subject to market risk. Please invest safely.</strong> <br><br><i>Recommendation</i>: <strong>BUY</strong>"
    elif(predicted_5_days_forecast_price_test_x.mean() < get_quote_data('btc-usd')['regularMarketPrice'] and p_perc < n_perc):
        final_result= "<strong>Most probably if you short this crypto currency you will make profit. Algorithm is subject to market risk. Please invest safely.</strong> <br><br><i>Recommendation</i>: <strong>SELL</strong>"
    else:
        if(recommendation_score>9.5):
            final_result= "<strong>Most probably if you buy this crypto currency you will make profit. Algorithm is subject to market risk. Please invest safely.</strong> <br><br><i>Recommendation</i>: <strong>BUY</strong>"
        else:
            final_result= "<strong>Most probably if you short this crypto currency you will make profit. Algorithm is subject to market risk. Please invest safely.</strong> <br><br><i>Recommendation</i>: <strong>SELL</strong>"
    
    
    final_score = finalScore(recommendation_score)
    
    five_days_pred = []
    for i in predicted_5_days_forecast_price_test_x:
      five_days_pred.append(i)
    
    return render_template('predict.html', coin=coin, mean_predicted=mean_predicted, positive_tweets=round(p_perc,2), negative_tweets=round(n_perc,2), score=final_score, final_result=final_result, v1=five_days_pred[0],v2=five_days_pred[1],v3=five_days_pred[2],v4=five_days_pred[3],v5=five_days_pred[4])

def dataset_generator_lstm(dataset, look_back=5):
    dataX, dataY = [], []
    
    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0]) # this is the label or actual y-value
    return np.array(dataX), np.array(dataY)

# Sentiment analysis functions 

def connect():
        consumer_key = 'YOUR_consumer_key'
        consumer_secret = 'YOUR_consumer_secret'
        access_token = 'YOUR_access_token'
        access_token_secret = 'YOUR_access_token_secret'

        try:
            auth = OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            api = tweepy.API(auth)
            return api
        except:
            print("Error")
            exit(1)

def cleanText(text):
  text = text.lower()
  # Removes all mentions (@username) from the tweet since it is of no use to us
  text = re.sub(r'(@[A-Za-z0-9_]+)', '', text)
    
  # Removes any link in the text
  text = re.sub('http://\S+|https://\S+', '', text)

  # Only considers the part of the string with char between a to z or digits and whitespace characters
  # Basically removes punctuation
  text = re.sub(r'[^\w\s]', '', text)

  # Removes stop words that have no use in sentiment analysis 
  text_tokens = word_tokenize(text)
  text = [word for word in text_tokens if not word in stopwords.words()]

  text = ' '.join(text)
  return text

def stem(text):
  # This function is used to stem the given sentence
  porter = PorterStemmer()
  token_words = word_tokenize(text)
  stem_sentence = []
  for word in token_words:
    stem_sentence.append(porter.stem(word))
  return " ".join(stem_sentence)

def sentiment(cleaned_text):
  # Returns the sentiment based on the polarity of the input TextBlob object
  if cleaned_text.sentiment.polarity > 0:
    return 'positive'
  elif cleaned_text.sentiment.polarity < 0:
    return 'negative'
  else:
    return 'neutral'

def fetch_tweets(query, count = 300):
  api = connect() # Gets the tweepy API object
  tweets = [] # Empty list that stores all the tweets
  
  try:
    # Fetches the tweets using the api
    fetched_data = api.search_tweets(q = query + ' -filter:retweets', count = count)
    for tweet in fetched_data:
      txt = tweet.text
      clean_txt = cleanText(txt) # Cleans the tweet
      stem_txt = TextBlob(stem(clean_txt)) # Stems the tweet
      sent = sentiment(stem_txt) # Gets the sentiment from the tweet
      tweets.append((txt, clean_txt, sent))
    return tweets
  except Exception as e:
    print("Error : " + str(e))
    exit(1)

def finalScore(recommendation_score):
  score = round(recommendation_score,2)
  score = (score + 100) / 2
  return score

if __name__=="__main__":
    app.run(debug=True)