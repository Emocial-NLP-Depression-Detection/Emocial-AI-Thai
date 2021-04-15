from pythainlp.sentiment import sentiment
import twint
import pandas as pd
import os
c = twint.Config()
c.Search = "โรคซึมเศร้า"
c.Limit = 100000
c.Store_csv = True
c.Output = "./data/raw_data.csv"
c.Lang = "th"
os.remove("./data/raw_data.csv")
twint.run.Search(c)


# def translate(x):
#     return TextBlob(x).translate(to="th")
print("\nImporting Data\n")
df = pd.read_csv("./data/raw_data.csv")
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

print("\n Finished Sentiment Analysing the tweets\n")
df2 = pd.DataFrame({'Tweets': df['tweet'], 'label': label})

print("\nStart Cleaning..\n")
for index, i in df2.iterrows():
    # print(i)
    if df2.label.value_counts()[1] != df2.label.value_counts()[0] and i['label'] == 1:
        df2.drop(index, inplace=True)
print("\Done Cleaning..\n")
print(f"Depressed Tweets: {df2.label.value_counts()[1]}")
print(f"Positive Tweets: {df2.label.value_counts()[0]}")

df2.to_csv("./data/data.csv", index=False)
