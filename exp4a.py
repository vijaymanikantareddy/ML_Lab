# Linear Regression

import numpy as np 
import matplotlib.pyplot as plt

x = 2 * np.random.rand(100,1) 
y = 4 + 3 * x+ np.random.rand(100,1)
 
from sklearn.linear_model import LinearRegression 
lr = LinearRegression() 
lr.fit (x,y) 
print ('Intercept value:',lr.intercept_) 
print ('Coefficient value:', lr.coef ) 

y_pred = lr.predict (x) 
plt.figure(figsize=(10,5)) 
plt.xlabel('x')
plt.ylabel('y') 
plt.scatter (x,y) 
plt.plot (x,y_pred, color='red')