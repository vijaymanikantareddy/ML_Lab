from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
scalar = StandardScaler()
scalar.fit(df)
scalar_data = scalar.transform(df)
pca = PCA(n_components=2)
pca.fit(scalar_data)
x_pca = pca.transform(scalar_data)
x_pca.shape
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='plasma')
plt.xlabel('First Pricipal Component')
plt.ylabel('Second Principal Component')
