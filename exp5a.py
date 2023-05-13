# 5.) Bias, Variance

from mlxtend.evaluate import bias_variance_decomp 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.utils import shuffle 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
 
data = pd.read_csv("5 student.csv") 
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] 
data.head() 
predict = "G3" 
x = np.array(data.drop([predict], 1)) 
y = np.array(data[predict]) 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2) 
print (xtrain.shape, ytrain.shape, xtest.shape, ytest.shape) 

lr = LinearRegression() 
lr.fit(xtrain, ytrain) 
y_pred = lr.predict(xtest)
 
mse, bias, variance = bias_variance_decomp (lr, xtrain, ytrain, xtest, ytest, loss='mse', num_rounds=150) 
print ("MSE : %.3f"%mse) 
print ("Average Bias : %.3f" %bias) 
print ("Average Variance : %.3f" %variance) 




