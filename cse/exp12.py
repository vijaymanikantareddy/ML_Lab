import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
import seaborn as sns
df = pd.read_csv("User_Data.csv")
print(df)
df.head()
df.tail()
df.Age.describe()
df.info()
df.Gender.value_counts()
sns.catplot(x="Purchased", y="EstimatedSalary",
 data=df, kind="box", aspect=1.5)
mtp.title("Boxplot for target vs proline")
mtp.show()



# Output:
#  User ID Gender Age EstimatedSalary Purchased
# 0 15624510 Male 19 19000 0
# 1 15810944 Male 35 20000 0
# 2 15668575 Female 26 43000 0
# 3 15603246 Female 27 57000 0
# 4 15804002 Male 19 76000 0
# .. ... ... ... ... ...
# 395 15691863 Female 46 41000 1
# 396 15706071 Male 51 23000 1
# 397 15654296 Female 50 20000 1
# 398 15755018 Male 36 33000 0
# 399 15594041 Female 49 36000 1
# [400 rows x 5 columns]
#  User ID Gender Age EstimatedSalary Purchased
# 0 15624510 Male 19 19000 0
# 1 15810944 Male 35 20000 0
# 2 15668575 Female 26 43000 0
# 3 15603246 Female 27 57000 0
# 4 15804002 Male 19 76000 0
#  User ID Gender Age EstimatedSalary Purchased
# 395 15691863 Female 46 41000 1
# 396 15706071 Male 51 23000 1
# 397 15654296 Female 50 20000 1
# 398 15755018 Male 36 33000 0
# 399 15594041 Female 49 36000 1
# count 400.000000
# mean 37.655000
# std 10.482877
# min 18.000000
# 25% 29.750000
# 50% 37.000000
# 75% 46.000000
# max 60.000000
# Name: Age, dtype: float64
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 400 entries, 0 to 399
# Data columns (total 5 columns):
# # Column Non-Null Count Dtype 
# --- ------ -------------- -----
# 0 User ID 400 non-null int64 
# 1 Gender 400 non-null object
# 2 Age 400 non-null int64 
# 3 EstimatedSalary 400 non-null int64 
# 4 Purchased 400 non-null int64 
# dtypes: int64(4), object(1)
# memory usage: 15.8+ KB
# Female 204
# Male 196
# Name: Gender, dtype: int64
