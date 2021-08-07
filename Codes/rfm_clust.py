import pandas as pd
import numpy as np
# Import KMeans
from sklearn.cluster import KMeans

df = pd.read_csv("G:/Statistics (Python)/Cases/Recency Frequency Monetary/rfm_data_customer.csv",
                 index_col=0)

df_select = df.drop('most_recent_visit',axis=1)

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
df_select_scaled=scaler.fit_transform(df_select)

clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2021)
    model.fit(df_select_scaled)
    Inertia.append(model.inertia_)
    
# Import pyplot
import matplotlib.pyplot as plt

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

# Create a KMeans instance with clusters: Best k model
model = KMeans(n_clusters=4,random_state=2021)
model.fit(df_select_scaled)
# Cluster Centroids
print(model.cluster_centers_)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(df_select_scaled)


clusterID = pd.DataFrame({'ClustID':labels},index=df_select.index)
clusteredData = pd.concat([df_select,clusterID],
                          axis='columns')

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')











