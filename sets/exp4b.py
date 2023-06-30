# Logistic Regression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_set = pd.read_csv("User_Data.csv")
x = data_set.iloc[:,[2, 3]].values
y = data_set.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix\n", cm)
print("Accuracy : ", accuracy_score(y_test, y_pred))


'''
# Output:
confusion matrix
[[79 6]
[11 38]]
Accuracy : 0.8731343283582089
'''