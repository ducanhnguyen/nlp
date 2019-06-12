'''
Convert terms into vector and plot them in two dimensional space.

The same meaning terms tend to be closer in the plot.
'''
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
nltk.download('stopwords')
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()

HEADLINE_COLUMN = "headline"
SHORT_DESCRIPTION_COLUMN = "short_description"
CATEGORY_COLUMN = "category"

def read_data():
    COLAB_PATH = '/content/drive/My Drive/Colab Notebooks/News_Category_Dataset_v2.json'
    LOCAL_PATH = r'../data/News_Category_Dataset_v2.json'
    USING_PATH = LOCAL_PATH

    if USING_PATH == COLAB_PATH:
        from google.colab import drive
        drive.mount('/content/drive/')

    articles = pd.read_json(USING_PATH, lines=True)
    articles = articles.drop(
        columns=['link', 'date', 'authors'], axis=0)

    print('Columns = ' + str(articles.columns))
    # print('Index = ' + str(reviews.index))

    limit = 10000
    articles = articles[:limit]
    print('Shape of articles = ' + str(articles.shape))
    return articles

def group_data(articles):
    group = dict()
    for i in range(len(articles)):
        category = articles.at[i, CATEGORY_COLUMN]
        content = articles.at[i, SHORT_DESCRIPTION_COLUMN] + '. '+articles.at[i, HEADLINE_COLUMN]+ '. '

        if category not in group:
            group[category] = content
        else:
            group[category] += content

    return group

def tokenizer(document):
    document = str(document).lower()
    document = re.sub('[^\w]', ' ', document)
    # [thing for thing in list_of_things]
    tokens = [token for token in document.split(' ') if len(token) >= 2 and token not in stopwords.words('english')]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return tokens

articles = read_data()

group = group_data(articles)
print('Categories = ' + str(group.keys()))

# tokenize
tokens_list = []
for k, v in group.items():
    tokens_list.append(tokenizer(v))

# mapping
word_index_map = dict()
count = 0

print('creating mapping from groups of articles')
for tokens in tokens_list:
    for token in tokens:
        if (token not in word_index_map):
            word_index_map[token] = count
            count += 1

print('Dictionary (word, index)+: size = ' + str(len(word_index_map)))
print('Vocabularies: ' + str(word_index_map.keys()))

# create term-document matrix: row is term, column is documment.
print('creating term-document matrix')
column = len(tokens_list)
row = len(word_index_map)

X = np.zeros(shape=(row, column))
columnId = -1
for tokens in tokens_list:
    columnId += 1
    for token in tokens:
        X[word_index_map[token], columnId] += 1

index_word_map = dict()
for item in word_index_map.items():
    index_word_map[item[1]] = item[0]

print('X shape: ' + str(X.shape))

# Dimensionality reduction to two dimension
svd = TruncatedSVD(n_components=2, random_state=42)
Z = svd.fit_transform(X)

plt.scatter(Z[:, 0], Z[:, 1])
plt.xlabel('x')
plt.ylabel('y')

for i in range(row):
    plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))

plt.show()