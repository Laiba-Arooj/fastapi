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
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
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
with open("model/svc_model.pkl", "rb") as f:
    model = joblib.load(f)
# load tfidf vectorizer
with open("vectorizer/tfidf_vec.pkl", "rb") as vect:
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
col = "Tweet Text"
def prepare(doc):
  new_doc = [clean_text(doc)]
  feat = tfidf.transform(new_doc)
  a = pd.DataFrame(feat.todense(), columns = tfidf.get_feature_names_out())

  return a

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
#   dfr = pd.DataFrame(data, columns=columns)
#   dfr.to_csv('tweets.csv', index=False)
    print(data)
data_pred= pd.read_csv("predictions.csv")
sizes = data_pred['Emotion'].value_counts()
sizes = np.array(sizes)
labels = data_pred['Emotion'].unique()
def pieC():
    data_pred= pd.read_csv("predictions.csv")
    plt.figure(figsize=(7,6))
    plt.pie(sizes, labels=labels,
            autopct='%.2f%%', shadow=True, startangle=100)
    plt.axis('equal')

    plt.savefig("Outputs/output_pie.jpg")
    print("Saved Image")
def barC():
    data_pred['Emotion'].value_counts().plot(kind="bar",figsize=(7, 6),rot=0)
    plt.xlabel("Emotions", labelpad=14)
    plt.ylabel("Emotion Count", labelpad=14)
    plt.title("Count of Emotions Predicted from Tweets", y=1.02)
    plt.savefig("Outputs/output_bar.jpg")
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
@app.get("/predict_keyword")
def predict_keyword(qr:str, count:int):
    tweets = []
    count = 1 
    for tweet in tweepy.Cursor(api.search_tweets, q = qr,count=count, lang = "en", result_type="mixed").items(20):
        count +=1
        try:
            data = [tweet.text]
            data = tuple(data)
            tweets.append(data)
        except tweepy.TweepyException as e:
            print(e)
            continue
        except StopIteration:
            break
    print(tweets)
    # df = pd.DataFrame(data, columns=col)
    # df.to_csv('tweets.csv', index=False)
    prd = []
    for i in tweets:
        #vec= tfidf.transform([out])
        preds = model.predict(prepare(i))
        print(preds)
        prd.append(preds)
    print(prd)
    col = ["Emotion"]
    dfr = pd.DataFrame(prd, columns = col)
    dfr.to_csv('predictions.csv', index= False)
    outval = []
    emotions = np.loadtxt("predictions.csv",dtype=str,encoding='UTF-8')
    for x in emotions:
        outval.append(x)
    print(outval)
    # for i in tweets:
    #     for j in outval:
    #         return{"Tweet":i, "Emotion":}
    return{"Tweets":tweets, "Emotion":outval[1:]}
@app.get("/get_trend")
def get_trends(woeid:int):
    trends=[]
    #woeid =2211096
    #fetching the trends
     # fetching the trends
    today_trend = api.get_place_trends(id = woeid)

    # printing the information
    for value in today_trend:
        for trend in value['trends'][:10]:
            trends.append(trend['name'])
        return{"Today's trends":trends}

# @app.get("/Barchart")
# async def bar_chart():
#     barC()
#     img = "Outputs/output_bar.jpg"
#     return FileResponse(img)

# @app.get("/piechart")
# async def pie_chart():
#     pieC()
#     img = "Outputs/output_pie.jpg"
#     return FileResponse(img)
