"""
Spam/Non-spam classifier

This code makes comparison between count vectorizer and tf-idf transformer.

Given a set of emails which are classified into two classes (i.e., spam, non-spam), we need to build a classifier.

First of all, we need to create an email-term matrix in which a term represents a word (i.e., not a phrase).
The value between a term and an email can be (1) the number of occurrences of that term in the given email,
(2) or tf-idf score. The first method is known as count vectorizer. The second one is tf-idf transformer.

Secondly, we need to train to build a classifier.

Dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset

Paper of dataset: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/doceng11.pdf
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def read_data(path):
    data = pd.read_csv(path, encoding="ISO-8859-1")

    data = data.drop(labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    data.columns = ['label', 'content']
    data['label'] = data['label'].replace('ham', 0)
    data['label'] = data['label'].replace('spam', 1)

    X = data['content']
    y = data['label']
    return X, y


def performCountVectorizer(X):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(X).toarray()
    return X


def performTfidfTransformer(X):
    X = performCountVectorizer(X)
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X).toarray()
    return X


def fit(X, y, method):
    X = method(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print('Method ' + str(method) + ' : accuracy on test set= ' + str(score))

    score = model.score(X_train, y_train)
    print('Method ' + str(method) + ' : accuracy on training set= ' + str(score))

X, y = read_data('../data/sms-spam-collection-dataset.csv')
fit(X, y, performTfidfTransformer)
fit(X, y, performCountVectorizer)
