# nlp
spam checking, review classifier, etc.


### Spam checking
Given a set of emails which are classified into two classes (i.e., spam, non-spam), we need to build a classifier.

*Dataset*: https://www.kaggle.com/uciml/sms-spam-collection-dataset

<img src="https://github.com/ducanhnguyen/nlp/blob/master/img/spam.png" width="950">

*Paper of dataset*: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/doceng11.pdf

This code makes comparison between count vectorizer and tf-idf transformer.

First of all, we need to create an email-term matrix in which a term represents a word (i.e., not a phrase).
The value between a term and an email can be (1) the number of occurrences of that term in the given email,
(2) or tf-idf score. The first method is known as count vectorizer. The second one is tf-idf transformer.

Secondly, we need to train to build a classifier.

Model: MultinomialNB (because we will use count vectorizer or tf-idf transformer, multinomial NB should be used)

Result: using count vectorizer is better than tf-idf transformer.

| vectorizer | test set | train set | 
| --- | --- | --- |
|count vectorizer| 0.98| 0.995
|tf-idf transformer| 0.968 | 0.981 | 

<img src="https://github.com/ducanhnguyen/nlp/blob/master/img/review.png" width="950">

<img src="https://github.com/ducanhnguyen/nlp/blob/master/img/lsa-0.png" width="950">
<img src="https://github.com/ducanhnguyen/nlp/blob/master/img/lsa-1.png" width="950">
<img src="https://github.com/ducanhnguyen/nlp/blob/master/img/lsa-2.png" width="950">
