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

