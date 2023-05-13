from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
x, y = datasets.load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=0)
k_folds = KFold(n_splits=5)
Scores = cross_val_score(clf, x, y, cv=k_folds)
print("cross validation scores:", Scores)
print("Average cv scores :", Scores.mean())
print("Number of cv scores used in Average", len(Scores))

# Output:
# cross validation scores: [1. 0.96666667 0.83333333 0.93333333 0.8 ]
# Average cv scores : 0.9066666666666666
# Number of cv scores used in Average 5
