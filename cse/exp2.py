import numpy as np
import pandas as pd
data = pd.read_csv("enjoysport.csv")
concepts = np.array(data.iloc[:, 0:-1])
print("Instances are : ", concepts)
target = np.array(data.iloc[:, -1])
print("Target values are : ", target)
def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("initialization of specific_h and general_h")
    print("Specific Boundary: ", specific_h)
    general_h = [["?" for i in range(len(specific_h))]
    for i in range(len(specific_h))]
    print("Generic Boundary: ", general_h)
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            print("Instance is Positive ")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "no":
            print("Instance is Negative ")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h


s_final, g_final = learn(concepts, target)
print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")


# Output:
# Instances are : 
# [['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
# ['Sunny' 'Warm' 'High' 'Strong' 'Warm' 'Same']
# ['Rainy' 'Cold' 'High' 'Strong' 'Warm' 'Change']
# ['Sunny' 'Warm' 'High' 'Strong' 'Cool' 'Change']]
# Target values are : ['yes' 'yes' 'no' 'yes']
# initialization of specific_h and general_h
# Specific Boundary: ['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
# Generic Boundary: [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], 
# ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
# Instance is Positive 
# Instance is Positive 
# Instance is Negative 
# Instance is Positive 
# Final Specific_h:
# ['Sunny' 'Warm' '?' 'Strong' '?' '?']
# Final General_h:
# [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]
