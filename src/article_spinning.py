import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')

wordnet_lemmatizer = WordNetLemmatizer()

REVIEW_COLUMN = "Text"
SCORE_COLUMN = 'Score'

def read_data():
    reviews = pd.read_csv('/Users/ducanhnguyen/Desktop/amazon-fine-food-reviews/Reviews.csv')
    reviews = reviews.drop(
        columns=['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time',
                 'Summary', 'Time', 'Summary'], axis=0)

    print('Columns = ' + str(reviews.columns))
    print('Index = ' + str(reviews.index))

    limit = None
    reviews = reviews[:limit]
    print('Shape of reviews = ' + str(reviews.shape))
    return reviews


def convert_score(reviews):
    for i in range(len(reviews[SCORE_COLUMN])):
        if reviews[SCORE_COLUMN][i] >= 3:
            reviews.at[i, SCORE_COLUMN] = 1  # positive reviews
        else:
            reviews.at[i, SCORE_COLUMN] = 0  # negative reviews

    return reviews


def normalize_text(reviews):
    for i in range(len(reviews[SCORE_COLUMN])):
        normalized_review = reviews.at[i, REVIEW_COLUMN].lower().replace('\n', '') \
            .replace(',', ' , ').replace('.', ' ') \
            .replace(')', ' ) ').replace('(', ' ( ') \
            .replace('  ', ' ').replace('<br />', '').replace('<br >', '').replace('!', ' ! ') \
            .replace("don't", 'do not').replace("'s", ' is').replace("'m", ' am').replace("can't", 'can not')
        reviews.at[i, REVIEW_COLUMN] = normalized_review

    return reviews


def review_tokenizer(document):
    # document = re.sub('[^\w]', ' ', document)
    # [thing for thing in list_of_things]
    tokens = [token for token in document.split(' ')]
    #tokens = [token for token in tokens if len(token) >= 1]
    tokens = [token for token in tokens if len(token) >= 1 and token not in ['!', ',', '.', '(', ')', '{', '}']]
    # tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def create_trigram(tokens_list):
    # dictionary of trigram
    # Example: {"I,you": {"love": 1, "hate": 1}}
    # "I": the previous token
    # "you": the next token
    # the middle token may be "love" (occur 1 time), "hate" (occur 1 time)
    trigrams = dict()

    for tokens in tokens_list:
        for i in range(1, len(tokens) - 1):
            pre = i - 1
            next = i + 1

            key = tokens[pre] + "," + tokens[next]
            if key not in trigrams:
                trigrams[key] = dict()

            value = tokens[i]
            if value not in trigrams[key]:
                trigrams[key][value] = 1
            else:
                trigrams[key][value] += 1

    return trigrams

def convert_trigram_to_probability(trigrams):
    # covert dictionary of trigram to probability
    # Example: {"I,you": {"love": 0.5, "hate": 0.5}}
    for key, value in trigrams.items():
        total = 0
        for k, v in value.items():
            total += v

        for k, v in value.items():
            value[k] = v / total
    return trigrams

def test(reviews, trigrams):
    review = reviews.at[30, REVIEW_COLUMN]

    tokens = review.split(' ')
    new_review = tokens[0]

    for i in range(1, len(tokens) - 1):
        pre = i - 1
        next = i + 1
        choice = tokens[i]

        key = tokens[pre] + "," + tokens[next]
        if key in trigrams:
            possible_choices = trigrams[key]

            if (np.random.random() > 0.5 and len(possible_choices.items()) >= 2):
                # multinomial sampler
                probs = []
                words = []
                for _, prob in possible_choices.items():
                    probs.append(prob)
                    words.append(_)

                sample = np.random.multinomial(n=10, size=1, pvals=probs)  # 1 experiment, 1 trial
                sample = np.argmax(sample)
                replacement = words[sample]

                if choice != replacement:
                    print('Trigram')
                    print('\t' + str(key))
                    print('\t' + str(possible_choices))
                    print('Replace: ' + str(choice) + ' -> ' + replacement)
                    print()

                new_review += ' ' + replacement
            else:
                new_review += ' ' + choice
        else:
            new_review += ' ' + choice

    new_review += ' ' + tokens[-1]

    print('Original: ' +  str(review))
    print('Modified: ' + str(new_review))

reviews = read_data()
convert_score(reviews)
normalize_text(reviews)

# tokenize
tokens_list = []
for i in range(len(reviews)):
    text = reviews.at[i, REVIEW_COLUMN]
    tokens_list.append(review_tokenizer(text))

#
trigrams = create_trigram(tokens_list)
trigrams = convert_trigram_to_probability(trigrams)

# test
test(reviews, trigrams)

