# Logistic Regression

from skleam import datasets
import numpy as np
import matplotlib.pyplot as plt 

iris = datasets.load_iris()
print (iris.keys())
 
print (iris["feature_names"]) 
iris
 
iris['target']
 
x = iris["data"][:,31]
y = (iris['target'] == 2).astype(np.int)
x.shape, y.shape

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression ()
log_reg.fit (x,y) 

print ('Logistic Regression Accuracy Score:', log_reg.score (x,y)) 
y_pred = log_reg.predict (x) 
y_pred 

from sklearn.metrics import confusion_matrix, accuracy_score 
print (confusion_matrix (y,y_pred)) 

accuracy_score(y,y_pred) 

x_new = np.linspace (0,3, 1000).reshape (-1, 1) 

y_proba = log(reg.predict_proba (x_new)
 
plt.plot (x_new, y_proba[:,1],"g", label = "Iris-Virginica") 
plt.plot (x_new, y_proba[:,0],"b--", label = "Not Iris-Virginica") plt.xlabel ("Petal width (cm)") 
plt.ylabel ("Probability") 
plt.legend()
