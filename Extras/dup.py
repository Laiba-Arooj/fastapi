from ast import keyword
import numpy as np
import io
import nltk
import pandas as pd
import re
import emot
from regex import P
import tweepy
import tweepy
import csv
import matplotlib.pyplot as plt
import string
import re
import joblib
from pydantic import BaseModel
from typing import List
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
nltk.download('wordnet')
from nltk import pos_tag
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from fastapi import FastAPI
nltk.download('omw-1.4')

# keys and tokens from the Twitter Dev Console
consumer_key = 'E7zqWoQ7tH7wRFTw3RNvqxgd1'
consumer_secret = 'vXrN9CotSGk79JgMnU0VDaDBMCwH2ARBfL5a6AmARnBiIZEfRq'
access_token = '1003798091122839552-V0fhM4t8Jw7lz6C3Ju7vBb0ZuAK03O'
access_token_secret = 'jJ0rR1mpEHO46TkJ2WmuaN9jfbl4hULz98aUWbpUhH0Wu'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# load the sentiment model
with open("model/sentiment_model_pipeline.pkl", "rb") as f:
    model = joblib.load(f)
# load tfidf vectorizer
with open("vectorizer/tfidf.pkl", "rb") as vect:
    tfidf = joblib.load(vect)



# return the wordnet object value corresponding to the POS tag
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
  
def convert_emoticons(text):
  for emot in UNICODE_EMOJI:
        text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
  return text
def clean_text(text):
    # lower text
    text = str(text).lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    additional  = ['rt','rts','retweet']
    stop = set().union(stopwords.words('english'),additional)
    stop.remove("not")

    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text (transform every word into their root form (e.g. rooms -> room, slept -> sleep))
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return(text)

def getTweets(query, count=7):
  fetched_tweets = api.search_tweets(q = query, count = count, lang="en",result_type ="recent", tweet_mode="extended", include_rts=0)
  columns = ['Tweet']
  data = []
  # parsing tweets one by one
  for tweet in fetched_tweets:
      # do something with standard tweets
      clean_text(tweet)
      data.append([tweet.full_text])
      tweet.retweeted_status.full_text if tweet.full_text.startswith("RT @") else tweet.full_text
  dfr = pd.DataFrame(data, columns=columns)
  dfr.to_csv('tweets.csv')

class Item(BaseModel):
    query: str
    

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict_sentiment")
def predict_sentiment(review:str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    text = [clean_text(review)]
    vec = tfidf.transform(text)
    prediction = model.predict(vec)
    print(prediction)
    if prediction == "anger":
        prediction="Anger"
    elif prediction == "fear":
        prediction = "Fear"
    elif prediction == "joy":
        prediction = "Joy"
    elif prediction == "love":
        prediction = "Love"
    elif prediction == "sadness":
        prediction = "Sadness"
    elif prediction == "surprise":
        prediction = "Surprise"
    else:
        prediction = "Sentiment"

    return {"sentence":review,"prediction":prediction}
@app.get("/tweetByLink")
def tweetByLink(link : str):
    twtId = link.split('/')[-1]
    twtId = twtId[:19]
    print(twtId)
    tweet = api.get_status(twtId)
    print(tweet.text)
    output = clean_text(tweet.text)
    print(output)
    vec= tfidf.transform([output])
    prediction = model.predict(vec)
    print(prediction)
    if prediction == "anger":
        prediction="Anger"
    elif prediction == "fear":
        prediction = "Fear"
    elif prediction == "joy":
        prediction = "Joy"
    elif prediction == "love":
        prediction = "Love"
    elif prediction == "sadness":
        prediction = "Sadness"
    elif prediction == "surprise":
        prediction = "Surprise"
    else:
        prediction = "Sentiment"
    return {"sentence":output,"prediction":prediction}
@app.get("/predict_hashtag")

def predict_hashtag(items: List[Item], query:str):
    tweets = getTweets(query, count = 7)
    file_CSV = open('tweets.csv')
    data_CSV = csv.reader(file_CSV)
    print(data_CSV)
    stwt = list(data_CSV)
    print(stwt)
    prd = []
    for i in stwt:
      out = clean_text(i)
      print(out)
      vec= tfidf.transform([out])
      preds = model.predict(vec)
      print(preds)
      prd.append(preds)
    print(prd)
    return {"sentence":stwt,"prediction":prd}

    # col = ["Emotion"]
    # dfr = pd.DataFrame(prd, columns=col)
    # dfr.to_csv('predict.csv', index= False)
    # data_pred= pd.read_csv("predict.csv")
   
    # if prd == "anger":
    #     prd="Anger"
    # elif prd == "fear":
    #     prd = "Fear"
    # elif prd == "joy":
    #     prd = "Joy"
    # elif prd == "love":
    #     prd = "Love"
    # elif prd == "sadness":
    #     prd = "Sadness"
    # elif prd == "surprise":
    #     prd = "Surprise"
    # else:
    #     prdn = "Sentiment"
    
    # col = ["Emotion"]
    # dfr = pd.DataFrame(prd, columns=col)
    # dfr.to_csv('predict.csv', index= False)
    # data_pred= pd.read_csv("predict.csv")
    # print(data_pred)
    # cn = data_pred['Emotion'].value_counts()
