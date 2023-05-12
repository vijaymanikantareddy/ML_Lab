import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

mnist["data"].shape
mnist["data"]

mnist["target"].shape
mnist["target"]

X = mnist["data"]
y = mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

pca = PCA(n_components=0.95)
pca.fit(X_train)

print("No. of features after reduction: ", pca.n_components_)

pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X_train)

X_recovered = pca.inverse_transform(X_reduced)

X_reduced.shape
X_recovered.shape