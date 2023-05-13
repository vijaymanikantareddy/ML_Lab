# 5.) Cross Validation

from sklearn import datasets 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import KFold, cross_val_score 

x, y = datasets.load_iris(return_X_y=True) 
print ("Size of independent and dependent features: ", x.shape, y.shape)
 
dtclf = DecisionTreeClassifier(random_state=30)
 
cv_feature = KFold(n_splits = 4) 
cv_scores = cross_val_score(dtclf, x, y, cv = cv_feature) 

print ("Cross Validation Scores: ", cv_scores) 
print ("Average CV Score: ", cv_scores.mean()) 
print ("Number of CV Scores used in Average: ", len(cv_scores)) 
