import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
x = boston.data
y = boston.target

columns = boston.feature_names

boston_df = pd.DataFrame(boston.data)
boston_df.columns = columns
boston_df.describe()


# Handling missing data:

# Drop all rows that contain null values
boston_df = boston_df.dropna()
print("Data set size after dropng rows that contain null values: ", boston_df.shape)


# Fill the missing value in AGE feature with 30
boston_df['AGE'] = boston_df['AGE'].fillna(30)



# HANDLING OUTLIERS

# Boxplot for PTRATIO
import seaborn as sns
sns.boxplot(x = boston_df['PTRATIO'])

# Scatter plot between AGE and ZN
import matplotlib.pyplot as plt
plt.scatter(boston_df['AGE'], boston_df['ZN'])
plt.xlabel('AGE')
plt.ylabel('ZN')
plt.show()

# Z-Score
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(boston_df))
print(z)
boston_df_zscore = boston_df[(z < 3).all(axis = 1)]
boston_df_zscore.shape



# Inter Quartile Range (IQR)
q1 = boston_df.quantile(0.25)
q3 = boston_df.quantile(0.75)
iqr = q3-q1 
print(iqr)


# Remove all the outliers in our dataset
boston_df_iqr = boston_df[~((boston_df < (q1-1.5*iqr)) | (boston_df > (q3+1.5*iqr))).any(axis=1)]
boston_df_iqr.shape



# UNDERSTANDING RELATIONSHIPS AND NEW INSIGHTS THROUGH PLOTS

# Correlation matrix
corr_matrix = boston_df.corr().round(2)
plt.figure(1, figsize=(10, 5))
sns.heatmap(data = corr_matrix, annot=True, cmap = 'BrBG')
