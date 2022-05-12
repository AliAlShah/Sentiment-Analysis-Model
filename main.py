from cgitb import text
from urllib import response
import pandas as pd
import pickle
from telegram.ext import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

API = "5398865408:AAHmaukRXmxfhhUtOMIbX-6AvorE_wldIKE"
data = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter30k_cleaned.csv")

tfid = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
x = data["twitts"]
x = tfid.fit_transform(x)
y = data["sentiment"]

model = LogisticRegression()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model.fit(x_train, y_train)

def train(i):
    best = 0
    best_history = []
    for n in range(i):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        model.fit(x_train, y_train)

        score = model.score(x_test, y_test)
        
        if score > best:
            best = score
            best_history.append(best)
            with open("savedmodel.pickle", "wb") as f:
                pickle.dump(model, f)
        print(n)
    print(best_history)
    print(best)
#train(i)

def predict(text):
    vector = tfid.transform([text])
    pickle_in = open("savedmodel.pickle", "rb")
    answer = pickle.load(pickle_in).predict(vector)
    
    if answer == [1]:
        return "Good Sentiment"
    else:
        return "Bad Sentiment"

def start_command(update, context):
    update.message.reply_text("Type anything to get started")

def handle_message(update, context):
    text = str(update.message.text)
    response = predict(text)

    update.message.reply_text(response)

updater = Updater(API, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start_command))
dp.add_handler(MessageHandler(Filters.text, handle_message))

updater.start_polling()
updater.idle()