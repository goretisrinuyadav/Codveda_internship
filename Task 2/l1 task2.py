import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("3) Sentiment dataset.csv")
print(data.shape)
print(data.columns)
print(data.info())
print(data.describe())

#Histogram
plt.figure()
plt.hist(data["Likes"])
plt.title("likes distribution")
plt.xlabel("likes")
plt.ylabel("frequency")

#Boxplot
plt.figure()
plt.boxplot(data["Retweets"])
plt.title("boxplot of Retweets")
plt.ylabel("Retweets")

#line chart
plt.figure()
data["Year"].value_counts().sort_index().plot(kind="line")
plt.title("posts by year")
plt.xlabel("year")
plt.ylabel("Number of posts")

#scatter polt
plt.figure()
plt.scatter(data["Hour"],data["Likes"])
plt.title("Likes vs Hour")
plt.xlabel("Hour")
plt.ylabel("Likes")
plt.show()