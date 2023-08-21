"""
Subtask 4.1: Choose one machine learning algorithm (e.g., Decision Trees, SVM, Neural Networks),
and explain how it works. Include details about its application, advantages, and disadvantages.

My Answer:
•	Decision Trees are a machine learning model that builds a tree with if-else logic.
It takes a subset from the main dataset, then asks if-else questions depending on the data.
For example, is the person older than 40 or not? It creates a branch under this question as an answer yes or no.
It divides the rest of the data which corresponds to the previous question.
After division,
it makes the same operation for each branch until there is no data or a user-determined border has been reached.

•	Decision Trees have applications in classification and regression.
If a time-series problem is converted to a regression problem, decision trees can solve it too.
Before 20 years,
decision trees were also used in image and signal processing but neural networks doing this job nowadays.

•	For advantages, we should say that decision trees don’t need to be preprocessed input data as neural networks do.
Normalization and scaling processes also are not necessary for decision trees.
Training times are better compared to complex neural network models.

•	For disadvantages, decision trees converge to underfitting while training vast amounts of data. For random forest,
10k trees is a practical limit to build a model. Beyond this number of tree models can’t learn more detail of data,
and the model should be switched to neural network.


Subtask 4.2: Implement a simple example of the chosen algorithm using Python.
You can use a well-known dataset like the Iris dataset or any other dataset you prefer.
Provide the code and a brief explanation of the steps you took.
"""

import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

# import some data to play with
iris = datasets.load_iris()

# creating the dataframe from numpy array
data = pd.DataFrame(iris.data)
data.columns = ['len1', 'len2', 'len3', 'len4']
data['target'] = iris.target

# training and test data creation
x_train, x_test, y_train, y_test = train_test_split(data[['len1', 'len2', 'len3', 'len4']], data['target'],
                                                    test_size=0.1)
# model creation
lgb = LGBMClassifier(n_estimators=20)
lgb.fit(x_train, y_train)
print('model accuracy:', lgb.score(x_test, y_test))