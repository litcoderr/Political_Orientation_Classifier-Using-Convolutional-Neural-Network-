from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

from textblob import TextBlob

#Authenticate
consumer_key=
consumer_secret= 

access_token=
access_token_secret=

class listener(StreamListener):
	def on_data(self,data):
		print data
		return True

	def on_error(self,status):
		print status


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

twitterStream = Stream(auth,listener())
twitterStream.filter(track=["obama"])
