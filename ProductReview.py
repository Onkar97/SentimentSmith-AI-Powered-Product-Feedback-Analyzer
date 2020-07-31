import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Hides warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
csv = "./1429_1.csv"
df = pd.read_csv(csv)
df.head(2)

def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return '1'
    elif (rating == 3) or (rating == 2) or (rating == 1):
        return '0'
# Add sentiments to the data
df["Sentiment"] = df["reviews.rating"].apply(sentiments)

df['reviews.text']=df['reviews.text'].astype(str)

def helpful(rating):
    if (rating > 1):
        return '1'
    return '0'
    
df["helpful"] = df["reviews.numHelpful"].apply(helpful)

df.to_csv("./ProductReviews.csv", header='true',index='false')

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
tokenizer=RegexpTokenizer(r'\w+')
en_stopwords=set(stopwords.words('english'))
ps=PorterStemmer()

def getStemmedReview(reviews_text):
    reviews_text=reviews_text.lower()
    reviews_text=reviews_text.replace("<br /><br />"," ")
    #Tokenize
    tokens=tokenizer.tokenize(reviews_text)
    new_tokens=[token for token in tokens if token not in  en_stopwords]
    stemmed_tokens=[ps.stem(token) for token in new_tokens]
    clean_review=' '.join(stemmed_tokens)
    return clean_review

df['reviews.text'].apply(getStemmedReview)
X_train = strat_train["reviews.text"]
Y_train = strat_train["reviews.rating"]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8',decode_error='ignore')
vectorizer.fit(X_train)
X_train=vectorizer.transform(X_train)

df.fillna(' ')
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, Y_train.astype('U'))

print("Score on training data is: "+str(model.score(X_train,Y_train.astype('U'))))

import joblib
joblib.dump(model,'model.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')
