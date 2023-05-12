import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("file.csv")
df1 = pd.DataFrame(data)
print(df1)

f1 = df1['Distance_Feature'].values
f2 = df1['Speeding_Feature'].values
X = np.matrix(list(zip(f1, f2)))

plt.plot()
plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title('Driver Dataset')
plt.ylabel('Speeding_feature')
plt.xlabel('Distance_Feature')
plt.scatter(f1, f2)
plt.show()

# KMeans Algorithm
# K = 3

kmeans_model = KMeans(n_clusters=3).fit(X)

kmeans_model.cluster_centers_

kmeans_model.transform(X)

colors = ['b', 'g', 'r']

markers = ['o', 'v', 's']

plt.xlim([0, 100])
plt.ylim([0, 50])
plt.title("K Means Clustering with K = 3")
for i, l in enumerate(kmeans_model.labels_):
    plt.plot(f1[i], f2[i], color = colors[l], marker = markers[l], ls = 'None')

centroids = kmeans_model.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], markers='x', s=100, linewidths=3, color = 'k', zorder = 10)
plt.show()