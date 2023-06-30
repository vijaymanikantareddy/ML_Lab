# Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_set = pd.read_csv("salary_data.csv")
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
model = LinearRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
x_pred = model.predict(x_train)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, x_pred, color="red")
plt.title("Salary vs Experience(Training Dataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary(In rupees)")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_train, x_pred, color="red")
plt.title("Salary vs Experience(Test Dataset)")
plt.xlabel("years of Experience")
plt.ylabel("Salary(In rupees)")
plt.show()
