import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

data = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter30k_cleaned.csv")

tfid = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
x = data["twitts"]
x = tfid.fit_transform(x)
y = data["sentiment"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

model = LinearSVC()
model.fit(x_train, y_train)

