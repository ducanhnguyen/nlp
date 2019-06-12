"""
Build a classifier to detect if a review is positive or negative.

Dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews
"""
import re

import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle

nltk.download('stopwords')

wordnet_lemmatizer = WordNetLemmatizer()

REVIEW_COLUMN = "Text"
SCORE_COLUMN = 'Score'


def read_data():
    reviews = pd.read_csv('../data/amazon-fine-food-reviews/Reviews.csv')
    reviews = reviews.drop(
        columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time',
                 'Summary', 'Time', 'Summary'], axis=0)

    print('Columns = ' + str(reviews.columns))
    # print('Index = ' + str(reviews.index))

    limit = 20000  # use None for the whole dataset
    reviews = reviews[:limit]
    print('Shape of reviews = ' + str(reviews.shape))
    return reviews


def convert_rating_to_sentiment(reviews):
    """
    Rating >=3 -> positive reviews
    Otherwise, it is classified as negative review
    :param reviews:
    :return:
    """
    for i in range(len(reviews[SCORE_COLUMN])):
        if reviews[SCORE_COLUMN][i] >= 3:
            reviews.at[i, SCORE_COLUMN] = 1  # positive reviews
        else:
            reviews.at[i, SCORE_COLUMN] = 0  # negative reviews

    return reviews


def review_tokenizer(document):
    document = re.sub('[^\w]', ' ', document.lower())
    # [thing for thing in list_of_things]
    tokens = [token for token in document.split(' ')]
    tokens = [token for token in tokens if len(token) >= 1]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def create_mapping(tokens_list):
    word_index_map = dict()
    count = 0

    print('creating mapping from reviews')
    for tokens in tokens_list:
        for token in tokens:
            if (token not in word_index_map):
                word_index_map[token] = count
                count += 1

    return word_index_map


def create_document_term_matrix(reviews, word_index_map):
    '''
    Count vectorizer
    :param reviews:
    :param word_index_map:
    :return:
    '''
    print('creating document-term matrix')
    row = len(reviews.index)
    column = len(word_index_map)

    X = np.zeros(shape=(row, column))
    rowId = -1
    for tokens in tokens_list:
        rowId += 1
        for token in tokens:
            X[rowId, word_index_map[token]] += 1

    y = reviews[SCORE_COLUMN]

    return X, y


if __name__ == '__main__':
    reviews = read_data()
    convert_rating_to_sentiment(reviews)

    # tokenize
    tokens_list = []
    for i in range(len(reviews)):
        text = reviews.at[i, REVIEW_COLUMN]
        tokens_list.append(review_tokenizer(text))

    # mapping
    word_index_map = create_mapping(tokens_list)
    print('Dictionary (word, index)+: num of words = ' + str(len(word_index_map)))
    print('Vocabularies: ' + str(word_index_map.keys()))

    # create document-term matrix
    X, y = create_document_term_matrix(reviews, word_index_map)
    print('X shape: ' + str(X.shape))
    print('y shape: ' + str(y.shape))

    # Training
    print('Training')
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print('accuracy on test set = ' + str(score))

    score = model.score(X_train, y_train)
    print('accuracy on training set = ' + str(score))
