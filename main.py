
# -*- coding: utf-8 -*-
"""
REAL-TIME TWITTER THREAD TOPIC EXTRACTOR
Created on Sat May 22 07:11:14 2021

@author: Faris Hamidi
"""
#IMPORT NECESSARY PACKAGES
import pandas as pd
import numpy as np
import threading

#IMPORT PACKAGES FOR TWEET STREAMING
import tweepy
import json
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

#IMPORT NLP PACKAGES
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim

np.random.seed(400)

#SETTING UP AUTHENTICATION FOR TWITTER API
consumer_key = 'muUTwpBe7XnvxvyT7lUiBWr11'
consumer_secret = 'HLyMVNsgGlFlPaveM4Jns4yX7EaZsE022wX4LMI4eaIh2mfePN'
access_token = '1248812219405488128-iYnOgTb0sYpLi9poD93Vcp8uvg9iqn'
access_secret = '6iFLxvtLIXPuDy0YufkcafkGjTLABxPIySrJxH2aZ6gk2'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_secret)
api = tweepy.API (auth)

#GLOBAL DATAFRAME TO STORE TWEETS
df = pd.DataFrame()

#STEMMER USED
stemmer = SnowballStemmer("english")

#TWEET LISTENER CLASS DECLARATION TO HANDLE TWEETS RECEIVE
class MyListener(StreamListener):
    
    def __init__(self, api = None):
        super(StreamListener, self).__init__()
        self.num_tweets = 0
        
    def on_data(self, data):
        global df
        try:
            tweet = json.loads(data)
            if (tweet['lang']) == "en":
                
                if 'extended_tweet' in tweet:           #ONLY ACCEPT LONG TWEET (THREADS)
                    temp = {'tweet' : tweet['extended_tweet']['full_text']}
                    df = df.append(temp, ignore_index = True)
                    self.num_tweets += 1
                    
                else:
                    pass
            
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        
        return True
    
    def on_error(self, status):
       print(status)
       return True
   
    def on_status(self, status):
        if status.retweeted_status == 'true' :
            return 
        print(status)

#LEMMATIZER FUNCTION
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

#ARTICLE PREPROCESSOR FUNCTION (STOPWORD REMOVER, LEMMATIZER AND TOKENIZER)
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

#FUNCTION FOR STREAMING TWEETS
def collectingTweet():
    tweet_listener = MyListener()
    mytwitter_stream = Stream(auth, tweet_listener)
    mytwitter_stream.filter(track = ['virus'])

#LDA FUNCTION
def performLDA():
    global df
    df_assigned = pd.DataFrame()        #STORE TWEETS AND EXTRACTED TOPICS
    
    print("Training Data...")
    
    #IMPORT DATASETS FOR TRAINING PURPOSE
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
    
    #PREPROCESSING THE TRAINING DATASETS
    processed_docs = []
    for doc in newsgroups_train.data:
        processed_docs.append(preprocess(doc))
    
    dictionary = gensim.corpora.Dictionary(processed_docs)                  #BUILD DICTIONARY FOR THE TRAINING DATASETS AND PAIR WITH ITS FREQUENCY
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)   #FILTER TOO FREQUENT AND TOO RARE WORDS/TOKENS
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model =  gensim.models.LdaMulticore(bow_corpus,                     #TRAINING THE MODEL
                                            num_topics = 20, 
                                            id2word = dictionary,                                    
                                            passes = 10,
                                            workers = 2)
    print("Done Training!")
    with open('topic.txt', 'w') as outfile:                                 #WRITE ALL TOPICS FROM THE MODEL TO OUTPUT FILE
        for idx, topic in lda_model.print_topics(-1):
            outfile.write("Topic: {} \nWords: {}".format(idx, topic ))
            outfile.write("\n")
    
    print("Topic have been updated to topic.txt file...")
    
    pointer = 0
    
    while True:
        if pointer < len(df.index):                                         #IF THE TWEET NOT EXTRACTED YET, EXTRACT ITS TOPIC
            topic_score = ''
            sent = df.iloc[pointer]['tweet']
            processed = preprocess(sent)
            bow_vector = dictionary.doc2bow(processed)
            print(str(pointer) + " Topics extracted.")
            
            for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
                topic_score += ("Score: {}\t Topic: {}\n".format(score, lda_model.print_topic(index, 5)))
            
            temp = {'Tweet': sent, 'Topic_Score': topic_score}
            df_assigned = df_assigned.append(temp, ignore_index = True)
            
            df_assigned.loc[[pointer]].to_csv('table_result.csv', index = False, header = False, mode = 'a')        #APPEND TO OUTPUT CSV FILE
                
            pointer += 1


#RUN THE 2 FUNCTIONS SIMULTANEOUSLY     
thread1 = threading.Thread(target=collectingTweet)
thread2 = threading.Thread(target=performLDA)

thread1.start()
thread2.start()