from numpy import result_type
import tweepy
from tweepy import OAuth1UserHandler
import pandas as pd

# keys and tokens from the Twitter Dev Console
consumer_key = 'E7zqWoQ7tH7wRFTw3RNvqxgd1'
consumer_secret = 'vXrN9CotSGk79JgMnU0VDaDBMCwH2ARBfL5a6AmARnBiIZEfRq'
access_token = '1003798091122839552-V0fhM4t8Jw7lz6C3Ju7vBb0ZuAK03O'
access_token_secret = 'jJ0rR1mpEHO46TkJ2WmuaN9jfbl4hULz98aUWbpUhH0Wu'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

tweets = []
count = 1 
query = input("Enter the query: ")
for tweet in tweepy.Cursor(api.search_tweets, q = query,count=10, lang = "en", result_type="recent").items(10):
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
col = ["Emotion"]
    # dfr = pd.DataFrame(prd, columns=col)
    # dfr.to_csv('predictions.csv', index= False)
df = pd.DataFrame(tweets, columns=col)
df.to_csv("scrap.csv", index=False)