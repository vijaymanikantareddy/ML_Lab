from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
data = asarray([['red',], ['green'], ['blue']])
print(data)
encoder = OneHotEncoder(sparse=False)
onehot = encoder.fit_transform(data)
print(onehot)

# Output:
# [['red']
# ['green']
# ['blue']]
# [[0. 0. 1.]
# [0. 1. 0.]
# [1. 0. 0.]]
