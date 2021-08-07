import pandas as pd

nutrient = pd.read_csv("nutrient.csv",
                   index_col=0)

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
nutrientscaled=scaler.fit_transform(nutrient)

nutrientscaled = pd.DataFrame(nutrientscaled,
                          columns=nutrient.columns,
                          index=nutrient.index)

# Import KMeans
from sklearn.cluster import KMeans

clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2021)
    model.fit(nutrientscaled)
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
model = KMeans(n_clusters=5,random_state=2021)

# Fit model to points
model.fit(nutrientscaled)

# Cluster Centroids
print(model.cluster_centers_)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(nutrientscaled)


clusterID = pd.DataFrame({'ClustID':labels},index=nutrient.index)
clusteredData = pd.concat([nutrient,clusterID],
                          axis='columns')

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')

###########################Heirarchical##################################

# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(nutrientscaled,method='average')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(nutrient.index),
           leaf_rotation=45,
           leaf_font_size=10,
)

plt.show()