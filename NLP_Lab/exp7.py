# Implement text classification using naïve bayes classifier and text blob library.

!pip install -U scikit-learn
!pip install -U textblob

import nltk
nltk.download('punkt')

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

# Sample training data
train_data = [
 ('I love this car.', 'positive'),
 ('This view is amazing.', 'positive'),
 ('I feel great!', 'positive'),
 ('I dislike this product.', 'negative'),
 ('This place is horrible.', 'negative'),
 ('I feel sad.', 'negative')
]
# Create the Naive Bayes classifier
classifier = NaiveBayesClassifier(train_data)

# Sample test data
test_data = [
 'I like this movie.',
 'This food is terrible.',
 'I am happy.'
]
# Classify the test data
for text in test_data:
    sentiment = classifier.classify(text)
    print(f'Text: {text}')
    print(f'Sentiment: {sentiment}\n')


'''
Output:
    Text: I like this movie.
    Sentiment: positive
    Text: This food is terrible.
    Sentiment: positive
    Text: I am happy.
    Sentiment: positive
'''