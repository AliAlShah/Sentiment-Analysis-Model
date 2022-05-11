import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

data = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/twitter-data/master/twitter30k_cleaned.csv")

