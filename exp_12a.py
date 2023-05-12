import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
x = boston.data
y = boston.target

columns = boston.feature_names

boston_df = pd.DataFrame(boston.data)
boston_df.columns = columns
boston_df.describe()

