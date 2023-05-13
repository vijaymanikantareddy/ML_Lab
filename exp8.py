import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import datasets

# Load Iris dataset

iris = datasets.load_iris()

#The following variables x contains the four features and y contains the class labels

x =iris.data
y = iris.target

print('sepal-length', 'sepal-width', 'petal-length', 'petal-width')
print (x)

print ('class: 0-Iris-Setosa, 1-Iris-Versicolour, 2-Iris-Virginica')
print (y)

# Split the dataset into train and test data in 70:30 ratio.

# The train data contains 105 samples and test data contains 45 samples
x_train, x_test, y_train, y_test = train_test_split(xy, test_size=0.3)
# Train the model using KNN classifier with neighbours =5

knnclf = KNeighborsClassifier (n_neighbors = 5)

knnclf.fit (x_train,y_train)

# Make predictions for test data

y_pred = knnclf.predict (x_test)
print('--------------------------------------------')
print('\nConfusion Matrix for Test data: \n', metrics.confusion_matrix(y_test, y_pred))
print('--------------------------------------------')
print('\nKNN Classification Report: \n', metrics.classification_report(y_test, y_pred))
print('--------------------------------------------')
print('Test data Accuracy of the KNN Classifiers is %0.2f'%metrics.accuracy_score(y_test, y_pred))
print('--------------------------------------------')



# Print both correct and wrong predictions

i=0

print('--------------------------------------------')

print('%0-25s %-25s %-235s' % ('Original Label, Predicted Label', 'Correct/Wrong'))

for label in y_test:
    print ('%-258 %-25s' % (label, y_pred[i]), end="")
    if label == y_pred[i]:
        print(' %0-25s' % ("Correct"))
    else:
        print(' %-25s' % ("Wrong"))
    i=i+1