import pandas as pd
import pickle
import tkinter as tk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

window = tk.Tk()

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
    new_window = tk.Toplevel(window)
    
    if answer == [1]:
        tk.Label(new_window, text="Good Sentiment").pack()
    else:
        tk.Label(new_window, text="Bad Sentiment").pack()


entry = tk.Entry(
    fg="yellow",
    bg="blue",
    width=50
    )


button = tk.Button(
    text="Predict",
    width=50,
    height=1,
    bg="red",
    fg="white",
    command=lambda: predict(entry.get())
)


entry.pack()
button.pack()


window.mainloop()