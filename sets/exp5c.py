# Bias and Variance

from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeClassifier
from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
x, y = iris_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True, stratify=y)
tree = DecisionTreeClassifier(random_state=123)
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
 tree, x_train, y_train, x_test, y_test, loss='0-1_loss', random_seed=123, num_rounds=1000)
print(f'Average Expected loss:{round(avg_expected_loss,4)}n')
print(f'Average Bias:{round(avg_bias,4)}n')
print(f'Average Variance:{round(avg_var,4)}n')

'''
Output:
Average Expected loss:0.0607n
Average Bias:0.0222n
Average Variance:0.0393n
'''