import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

data = pd.read_csv('history_customers.csv')

data = pd.get_dummies(data, columns=['GENDER', 'EDUCATION_LEVEL', 'MARITAL_STATUS', 'INCOME_CATEGORY', 'CARD_CATEGORY'])

data = data.drop(['STATUS'], axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

k = 20
neigh = NearestNeighbors(n_neighbors=k)
nbrs = neigh.fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)

# Urutkan jarak dari setiap titik ke tetangga ke-k dan gambarkan grafik
distances = np.sort(distances, axis=0)[:, k-1]
plt.plot(distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel(f'{k}-distance')
plt.show()

epsilon = 4
min_samples = 100
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
dbscan.fit(data_scaled)

cluster_labels = dbscan.labels_
data['CLUSTER'] = cluster_labels

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
plt.figure(figsize=(10, 8))
for label, color in zip(np.unique(cluster_labels), ['b', 'g', 'r', 'c', 'm', 'y', 'k']):
    if label == -1:
        # Plot noise (outliers) dengan titik hitam
        plt.scatter(data_pca[cluster_labels == label, 0], data_pca[cluster_labels == label, 1], c='k', marker='.', s=20, label='Noise')
    else:
        plt.scatter(data_pca[cluster_labels == label, 0], data_pca[cluster_labels == label, 1], c=color, marker='o', s=30, label=f'Cluster {label}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.title('DBSCAN Clustering')
plt.show()

cluster_labels = dbscan.labels_
data['CLUSTER'] = cluster_labels

# Pisahkan data berdasarkan Cluster yang terbentuk
at_risk_customers = data[data['CLUSTER'] == -1]
cluster_0 = data[data['CLUSTER'] == 0]
cluster_1 = data[data['CLUSTER'] == 1]
cluster_2 = data[data['CLUSTER'] == 2]

# Simpan hasil dalam file CSV
at_risk_customers.to_csv('export\cluster_noise.csv', index=False)
cluster_0.to_csv('export\cluster_0.csv', index=False)
cluster_1.to_csv('export\cluster_1.csv', index=False)
cluster_2.to_csv('export\cluster_2.csv', index=False)
