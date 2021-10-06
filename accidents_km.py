import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# load and filter data
acc = pd.read_csv(r"accidents.csv/accidents.csv")
acc_filtered = acc[acc.State == "UT"]
acc_lat_lon = acc_filtered[["StartLat", "StartLng"]]

# data exploration
print(len(acc_lat_lon))
print(acc_lat_lon.head())

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

sns.relplot(
    x="StartLat", y="StartLng", data=acc_lat_lon, height=6,
)
plt.show()

# find optimum number of centroids K using elbow method
Sum_of_squared_distances = []
K = range(1,10)
for num_clusters in K :
    print(f"training on {num_clusters} clusters")
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(acc_lat_lon)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,"bx-")
plt.xlabel("Values of K") 
plt.ylabel("Sum of squared distances/Inertia") 
plt.title("Elbow Method For Optimal k")
plt.show()

# train model with K centroids (3)
model = KMeans(n_clusters = 3, random_state=0).fit(acc_lat_lon)
print(model.cluster_centers_)

# use model to predict 
acc_lat_lon["Cluster"] = model.predict(acc_lat_lon)
print(acc_lat_lon.head())

# Show predictions
sns.relplot(
    x="StartLat", y="StartLng", hue="Cluster", data=acc_lat_lon, height=6,
)
plt.show()