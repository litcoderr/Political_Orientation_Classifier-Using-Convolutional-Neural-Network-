import json
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import os

def getdata(read_path,down_path,csv_name):
	tweets_data = []
	tweets_file = open(read_path, "r")
	for line in tweets_file:
	    try:
	        tweet = json.loads(line)
	        tweets_data.append(tweet)
	    except:
	        continue

	print len(tweets_data)
	pics = pd.DataFrame()
	pics['name'] = map(lambda tweet: tweet['user']['name'] if tweet['user']['name']!=None else None,tweets_data)
	pics['text'] = map(lambda tweet: tweet['text'], tweets_data)
	pics['url'] = map(lambda tweet: tweet['user']['profile_image_url_https'] if tweet['user']['profile_image_url_https'] != None else None,tweets_data)
	pics['follower_num'] = map(lambda tweet: tweet['user']['followers_count'] if tweet['user']['followers_count']!=None else None,tweets_data)
	pics = pics.sort_values(by='follower_num',ascending=False)
	print(pics)
	pics.to_csv(csv_name,encoding='utf-8')

	getimg(pics,down_path)


def getimg(data,img_path):
	os.chdir(img_path)
	file_count = len([f for f in os.walk(".").next()[2] if f[-4:] == ".jpg"])
	i=file_count+1
	for index,row in data.iterrows():
		name = str(i)+'.jpg'
		urllib.urlretrieve(row['url'], name)
		i=i+1

#run getting dataframe and download image
getdata('/Users/jameschee/Desktop/Programming/python/Fun/twitter/twitter_data_obama.txt','/Users/jameschee/Desktop/Programming/python/Fun/twitter/obama_raw_img','obama.csv')

# tweets = pd.DataFrame()

# tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
# tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
# tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)

# tweets_by_lang = tweets['lang'].value_counts()

# fig, ax = plt.subplots()
# ax.tick_params(axis='x', labelsize=15)
# ax.tick_params(axis='y', labelsize=10)
# ax.set_xlabel('Languages', fontsize=15)
# ax.set_ylabel('Number of tweets' , fontsize=15)
# ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
# tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')
# plt.show()

# tweets_by_country = tweets['country'].value_counts()

# fig, ax = plt.subplots()
# ax.tick_params(axis='x', labelsize=15)
# ax.tick_params(axis='y', labelsize=10)
# ax.set_xlabel('Countries', fontsize=15)
# ax.set_ylabel('Number of tweets' , fontsize=15)
# ax.set_title('Top 5 countries', fontsize=15, fontweight='bold')
# tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')
# plt.show()