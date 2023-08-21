"""
Subtask 5.1: Explain the importance of data preprocessing and feature engineering in machine learning.
Discuss common techniques such as normalization, scaling, handling missing values, and encoding categorical variables.

My Answer:
There is a diagram on the internet
that shows most of the effort in a data science project belongs to preprocessing and cleaning operations.
And modeling is the smallest part of projects. In real life,
the data is never ready for modeling and need to be processed to a convenient model.
Feature engineering is the next part after preprocessing in data science projects.
We generate new features from data or merge new features from outsources.
This operation is a need to build stronger models.
For example, in energy consumption prediction,
we use not only the historical consumption data but also the weather data as well.
Because the consumption can vary depending on temperature, rain, snow, or wind speed.
In preprocessing, we normalize the numerical data. Because most of our models use linear methods in the backend.
If the data has bad distribution, this harms the model equations and gives poor learning scores.
So, we need data with convenient standard deviation and variance.
Scaling is also an important process because of the same reasons as normalization.
Generally, the 0-1 range is good for scaling the data, especially in deep learning.
Our data has missing values all the time because of many reasons.
There could be a data loss, or the data-collecting device may be out of service for that time.
We can fill these missing values with lots of methods or drop these samples from the data simply.
The data can be numerical and categorical as well.
For categorical data, we encode the data as many as the number of categories.
For linear-based models, this is a must. For nonlinear models like decision trees, encoding is not a must.

Subtask 5.2: Choose a dataset (real or hypothetical),
and describe how you would preprocess it for a specific machine learning task.
Include details on the techniques you would apply and why.

My Answer:
I chose the iris dataset. This dataset includes eye colors and four axes length of these eyes.
I will use multilayer perceptron aka neural networks classifier for the classification task.
The data is cleaned already. This means the cleaning and missing value imputation processes will be skipped.
We have four numerical features and one categorical target variable which has three classes.
First, I scaled the numerical features in to a range of zero and one.
Because I will use neural network, and they are so sensitive to beyond 0-1 range.
I want to improve my learning accuracy. Standard deviation and variance should be adjusted,
and there comes the standard scaler to help. It makes our standard deviation 1 and mean 0.
In neural networks, categorical variables should be converted into one-hot encoded format.
I used one-hot encoder for this task.
I divided my dataset into train and test datasets to evaluate the model better.
Finally, MLPClassifier from scikit-learn helps us to build the model.


Subtask 5.3: Implement a Python code snippet that demonstrates one or more preprocessing techniques on a sample dataset.
You can use libraries like Scikit-learn for this purpose.
"""
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