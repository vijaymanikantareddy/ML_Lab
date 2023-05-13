import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
data_set = pd.read_csv("Iris.csv")
x = data_set.iloc[:, [2, 3]].values
y = data_set.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
 x, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Output:
# ['Iris-virginica' 'Iris-versicolor' 'Iris-setosa' 'Iris-virginica'
# 'Iris-setosa' 'Iris-virginica' 'Iris-setosa' 'Iris-versicolor'
# 'Iris-versicolor' 'Iris-versicolor' 'Iris-virginica' 'Iris-versicolor'
# 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-setosa'
# 'Iris-versicolor' 'Iris-versicolor' 'Iris-setosa' 'Iris-setosa'
# 'Iris-virginica' 'Iris-versicolor' 'Iris-setosa' 'Iris-setosa'
# 'Iris-virginica' 'Iris-setosa' 'Iris-setosa' 'Iris-versicolor'
# 'Iris-versicolor' 'Iris-setosa' 'Iris-virginica' 'Iris-versicolor'
# 'Iris-setosa' 'Iris-virginica' 'Iris-virginica' 'Iris-versicolor'
# 'Iris-setosa' 'Iris-versicolor']
# [[13 0 0]
# [ 0 16 0]
# [ 0 0 9]]
