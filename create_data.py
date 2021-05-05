from pythainlp.sentiment import sentiment
import twint
import pandas as pd
import os
import emoji
import re
c = twint.Config()
c.Search = "โรคซึมเศร้า"
c.Limit = 100000
c.Store_csv = True
c.Output = "./data/raw_depressed_data.csv"
c.Lang = 'th'
p = twint.Config()
p.Search = "เรา"
p.Limit = 100000
p.Store_csv = True
p.Output = "./data/raw_everyday_data.csv"
p.Lang = 'th'

twint.run.Search(c)
twint.run.Search(p)



# def translate(x):
#     return TextBlob(x).translate(to="th")
print("\nImporting Data\n")
df = pd.read_csv("./data/raw_depressed_data.csv")
pos = pd.read_csv("./data/raw_everyday_data.csv")
print("\nDone Importing Data\n")
label = []
analysed = 0
print("\n Sentiment Analysing the tweets\n")
for tweets in df['tweet']:
    if sentiment(tweets) == 'pos':
        label.append(0)
    else:
        label.append(1)
    analysed = analysed + 1
    if analysed % 100 == 0:
        print(f"Analysed {analysed} so far...")

pos_label = []
for tweets in pos['tweet']:
    if sentiment(tweets) == 'pos':
        pos_label.append(0)
    else:
        pos_label.append(1)
    analysed = analysed + 1
    if analysed % 100 == 0:
        print(f"Analysed {analysed} so far...")

print("\n Finished Sentiment Analysing the tweets\n")
df2 = pd.DataFrame({'Tweets': df['tweet'], 'label': label})
pos2 = pd.DataFrame({'Tweets': pos['tweet'], 'label': pos_label})

is_pos = pos2['label'] == 0
pos2 = pos2[is_pos]
df3 = df2.append(pos2, ignore_index=True, sort=True)

print("\nStart Cleaning..\n")
for index, i in df3.iterrows():
    # print(i)
    if df3.label.value_counts()[1] > df3.label.value_counts()[0] and i['label'] == 1:
        df3 = df3.drop(index, errors='ignore')
        # print(f"Index: {index}\n I: {i}")
    elif df3.label.value_counts()[1] < df3.label.value_counts()[0] and i['label'] == 0:
        df3 = df3.drop(index, errors='ignore')
print("\Done Cleaning..\n")
print(f"Depressed Tweets: {df3.label.value_counts()[1]}")
print(f"Positive Tweets: {df3.label.value_counts()[0]}")

df3.to_csv("./data/data.csv", index=False)
os.remove("./data/raw_depressed_data.csv")
os.remove("./data/raw_everyday_data.csv")

