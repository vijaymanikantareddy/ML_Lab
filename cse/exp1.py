import pandas as pd
import numpy as np
data = pd.read_csv("https://github.com/manirevu10/MachineLearningLab/blob/master/enjoysport.csv")
print(data)
d = np.array(data.iloc[:, 0:-1])
print("The attributes are: ", d)
t = np.array(data.iloc[:, -1])
print("The target is: ", t)
def train(c, t):
    specific_h = c[0].copy()
    for i, val in enumerate(t):
        if val == 'yes':
            specific_h = c[i].copy()
            break
    for i, val in enumerate(c):
        if t[i] == 'yes':
            for j in range(len(specific_h)):
                if val[j] != specific_h[j]:
                    specific_h[j] = '?'
                else:
                    pass
    return specific_h

print("The Final hypothesis is : ",train(d, t))



# Output:
#  Sky AirTemp Humidity Wind Water Forecast EnjoySport
# 0 Sunny Warm Normal Strong Warm Same yes
# 1 Sunny Warm High Strong Warm Same yes
# 2 Rainy Cold High Strong Warm Change no
# 3 Sunny Warm High Strong Cool Change yes
# The attributes are: [['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
# ['Sunny' 'Warm' 'High' 'Strong' 'Warm' 'Same']
# ['Rainy' 'Cold' 'High' 'Strong' 'Warm' 'Change']
# ['Sunny' 'Warm' 'High' 'Strong' 'Cool' 'Change']]
# The target is: ['yes' 'yes' 'no' 'yes']
# The Final hypothesis is : ['Sunny' 'Warm' '?' 'Strong' '?' '?']
