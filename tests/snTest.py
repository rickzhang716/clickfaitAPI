import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
data_url = "https://bit.ly/3cngqgL" # or "path/to/biden_trump_tweets.csv"
df = pd.read_csv(data_url, 
                 parse_dates=['date_utc'], 
                 dtype={'hour_utc':int,'minute_utc':int,'id':str}
                )


g = df.groupby(['hour_utc','minute_utc','username'])
tweet_cnt = g.id.nunique()
tweet_cnt.head()
jb_tweet_cnt = tweet_cnt.loc[:,:,'JoeBiden'].reset_index().pivot(index='hour_utc', columns='minute_utc', values='id')
jb_tweet_cnt.fillna(0, inplace=True)

# Ensure all hours in table
jb_tweet_cnt = jb_tweet_cnt.reindex(range(0,24), axis=0, fill_value=0)
# Ensure all minutes in table
jb_tweet_cnt = jb_tweet_cnt.reindex(range(0,60), axis=1, fill_value=0).astype(int) 

dt_tweet_cnt = tweet_cnt.loc[:,:,'realDonaldTrump'].reset_index().pivot(index='hour_utc', columns='minute_utc', values='id')
dt_tweet_cnt.fillna(0, inplace=True)
dt_tweet_cnt = dt_tweet_cnt.reindex(range(0,24), axis=0, fill_value=0)
dt_tweet_cnt = dt_tweet_cnt.reindex(range(0,60), axis=1, fill_value=0).astype(int)

sns.heatmap(dt_tweet_cnt)
plt.show()