import pandas as pd

msg = pd.read_csv('filename.csv', names=['message', 'label'])

print('The dimension of the dataset', msg.shape)

msg['labelnum'] = msg.label.map({'pos':1, 'neg': 0})
X = msg.message
y = msg.labelnum
print(X)
print(y)


# Splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y)

print('\n The total number of Training Data: ', ytrain.shape)
print('\n The total number of Test Data: ', ytest.shape)


# Output of count vectorizer is a sparse matrix

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)
print('\n The words or Tokens in the text documents \n')
print(count_vect.get_feature_names())

xtrain
xtrain_dtm

df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names())

df.head(12)

# Training Naive Bayes Classifier on training data.

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm, ytrain)
train_predicted = clf.predict(xtrain_dtm)
test_predicted = clf.predict(xtest_dtm)

# Printing accuracy, Confusion matrix, Precision and Recall
from sklearn import metrics
print('\n Accuracy of the classifier is for Train data: ', metrics.accuracy_score(ytrain, train_predicted))
print('\nAccuracy of the classifier is for Test data: ', metrics.accuracy_score(ytest, test_predicted))

print('\n Confusion matrix for Train Data')
print(metrics.confusion_matrix(ytrain, train_predicted))
print('\nThe value of Precision', metrics.precision_score(ytrain, train_predicted))
print('\nThe value of Recall', metrics.recall_score(ytrain, train_predicted))

print('\n Confusion matrix for Test data')
print(metrics.confusion_matrix(ytest, test_predicted))
print('\n The value of Precision', metrics.precision_score(ytest, test_predicted))
print('\n The value of Recall', metrics.recall_score(ytest, test_predicted))

