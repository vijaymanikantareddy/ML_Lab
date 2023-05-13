# 5.) Remove duplicates

import pandas as pd 
data ={'F1': [1, 1, 1, 2], 'F2': [2, 2, 2, 3], 'F3': [3, 3, 4, 5]} 
df = pd.DataFrame(data) 
print ('Source DataFrame:\n', df)
 
result_df = df.drop_duplicates() 
print ('Result DataFrame:\n', result_df) 

result_df = df.drop_duplicates(keep=False) 
print ('Result DataFrame:\n', result_df)
 
result_df = df.drop_duplicates(subset=['F1', 'F2']) 
print ('Result DataFrame:\n', result_df) 

result_df = df.drop_duplicates(subset=['F1', 'F2'], keep='last') 
print ('Result DataFrame:\n', result_df) 
