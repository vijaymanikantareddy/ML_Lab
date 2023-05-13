import csv
a = []
with open('enjoysport.csv','r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)
    print(a)

print("\n The Total Number of Training Instances are: ", len(a))
num_attribute = len(a[0])-1
print("\n The initial hypothesis is: ")
hypothesis = ['0']*num_attribute
print(hypothesis)

for i in range(1, len(a)):
    if a[i][num_attribute] != 'No':
        print('\n Vector {} instance is: '.format(i), a[i])
        for j in range(0, num_attribute):
            if  hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'
    print("\n The hypothesis for the training instance {} is : \n".format(i+1), hypothesis)

print("\n The Maximally specific hypothesis for the training instance is ")
print(hypothesis)


