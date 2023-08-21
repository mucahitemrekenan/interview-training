import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier


# import some data to play with
iris = datasets.load_iris()

# creating the dataframe from numpy array
data = pd.DataFrame(iris.data)
features = ['len1', 'len2', 'len3', 'len4']
data.columns = features
data['target'] = iris.target

# minmax normalization
mm_scaler = MinMaxScaler(feature_range=(0, 1))
data[features] = mm_scaler.fit_transform(data[features])

# standard scaling of data
s_scaler = StandardScaler()
data[features] = s_scaler.fit_transform(data[features])

# one hot encoding of data
oh_encoder = OneHotEncoder()
target = oh_encoder.fit_transform(data[['target']]).toarray()

# training and test data creation
x_train, x_test, y_train, y_test = train_test_split(data[['len1', 'len2', 'len3', 'len4']], target, test_size=0.1)

# model creation
clf = MLPClassifier(hidden_layer_sizes=(10, 20, 10), random_state=1, max_iter=300).fit(x_train, y_train)
print('model accuracy:', clf.score(x_test, y_test))


lgb = LGBMClassifier(n_estimators=20)
lgb.fit(x_train, y_train)
print('model accuracy:', lgb.score(x_test, y_test))