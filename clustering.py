"""
Subtask 6.1: Explain the concepts of clustering methods, specifically K-means and Hierarchical clustering.
Discuss their advantages and disadvantages, and where they might be used.

My Answer:
Clustering is a method that divides the samples into groups depending on their similarities.
These similarities are calculated with distance, density, hierarchical, grid-based, neural network based, etc.
Clustering methods are used in grouping the customers of a company like Netflix, Facebook, and Google.
Netflix recommends new series to the customers depending on their calculated group.
Facebook and Google show related ads to the customer.
K-Means, Hierarchical, and DBSCAN are some of the clustering methods.
K-Means is a distance-based method. It selects K random centers and assigns the nearest sample points to that cluster.
Then calculates each cluster center again by meaning the sample points,
and assigns the nearest sample points to new cluster centers.
These center calculation and point assignment continue until the centers don’t move.
Hierarchical clustering uses dendrogram, a tree divides the data based on distance.
It splits the data into two with one of these distance metrics Euclidian, Manhattan, and cosine.
Then split the clusters again and again until there is no rest of the data can be split.
It has also linkage criteria for data points in a cluster. Single, complete, average,
and ward linkage determines data density in a cluster.
DBSCAN has two parameters epsilon and min points. Epsilon is the radius of the neighborhood around a data point.
Minpts are the minimum number of points required to form a dense region.
DBSCAN detects core points randomly which meet the epsilon and Minpts criteria.
Then expands the cluster until border points.
A border point is a point that meets epsilon criteria while it doesn’t meet the Minpts criteria.
All the connected core points are merged in a cluster. These steps are repeated until all core points have been visited.

Subtask 6.2: Choose a real or hypothetical problem and describe how you would apply clustering methods to solve it.
Include details on the specific algorithms you would use and why.

My Answer:
We select the diabetes data in scikit-learn to apply the clustering methods of K-means and DBSCAN.
I selected these methods because k-means calculate clusters with distance while DBSCAN uses linkage,
and density ignoring noise points.
Our data has different parameters from 414 diabetes patients.
All the parameters are numerical and scaled to 0-1.
This is important because the scale differences between features affect the DBSCAN algorithm negatively.
K-means is more robust to scale differences, but it has been affected as well.
We trained the scaled data with DBSCAN and K-means respectively.
According to plots, it is obvious that DBSCAN can’t split accurately the target variable into groups.
But the K-means split the target better.

Subtask 6.3: Implement a Python code snippet that demonstrates the use of a clustering method on a sample dataset.
You can use libraries like Scikit-learn for this purpose.
"""

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns


# import some data to play with
diabetes = datasets.load_diabetes(as_frame=True, scaled=True)
data = diabetes.data.copy()
data['target'] = diabetes.target


# this function creates clusters with selected features and clustering method.
def make_clusters(cluster_data, cluster_obj, features, method):
    # we select features to calculate clusters
    labels = cluster_obj.fit_predict(cluster_data[features])
    cluster_data['cluster_labels'] = labels

    # Create a scatter plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='cluster_labels', y='age', hue='target', palette='viridis', data=cluster_data)
    plt.title(f'{method}')
    plt.show()


clustering_features = ['sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# we call function for DBSCAN and KMEANS
make_clusters(data, DBSCAN(eps=0.1, min_samples=20, metric='euclidean'), clustering_features, 'DBSCAN')
make_clusters(data, KMeans(n_clusters=8), clustering_features, 'KMEANS')

