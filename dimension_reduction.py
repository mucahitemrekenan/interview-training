"""
Subtask 7.1: Explain the concepts of dimensionality reduction techniques,
specifically Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).
Discuss their advantages and disadvantages, and where they might be used.

My Answer:
Principal Component Analysis reduces the dimension of the data while retaining trends and patterns.
It standardizes the data first. Then the covariance matrix is calculated to capture the relationship between features.
With this covariance matrix, the eigenvalue decomposition is made.
After the decomposing process, we yield the eigenvalues and eigenvectors.
The eigenvectors are sorted by their eigenvalues in decreasing order.
According to the desired dimension space, the biggest eigenvalues are selected.
Finally, it projects the original data into lower-dimensional space by multiplying the data with the eigenvector matrix.
PCA reduces noise by keeping principal components with significant variance.
It reduces the dimension while retaining the most information, therefore, it improves the model performance.
PCA is a linear method because of this, it can’t catch non-linear relationships in the data.
PCA transforms the data into a new format after that interpretation of the data is challenging.

T-SNE algorithm also reduces the high dimensional data into low dimension.
It takes the high dimensional data
and calculates pairwise similarities between the samples with Gaussian probability density.
After that, t-SNE generates a random low-dimensional copy of the data,
and calculates the same probability for the low dimension.
With the help of Kullback-Leibler divergence, the loss value is calculated to update the low-dimensional data.
After several iterations, the algorithm will converge, and you will have low-dimensional data.
T-SNE transforms the data. The transformed data is hard to interpret as in PCA.
It is a non-convex algorithm because of these several iterations are needed to obtain the best low-dimensional data.


Subtask 7.2:
Choose a real or hypothetical problem and describe how you would apply dimensionality reduction techniques to solve it.
Include details on the specific algorithms you would use and why.

My Answer:
I created a code example which uses scikit-learn digits dataset.
I applied PCA and t-SNE algorithms to reduce dimensions. Finally, I plot the results side by side for comparison.
When you run the code, you will see two plots. The left shows the 2D representation of the dataset using PCA,
and the right plot shows the 2D representation using t-SNE.
T-SNE does a better job of separating the clusters corresponding to different digits compared to PCA.

Subtask 7.3:
Implement a Python code snippet that demonstrates the use of a dimensionality reduction technique on a sample dataset.
You can use libraries like Scikit-learn for this purpose.
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Apply PCA and reduce to two dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE and reduce to two dimensions
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plot PCA results
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
legend1 = ax1.legend(*scatter.legend_elements(), title="Digits")
ax1.add_artist(legend1)
ax1.set_title('PCA of Digits Dataset')

# Plot t-SNE results
scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
legend2 = ax2.legend(*scatter.legend_elements(), title="Digits")
ax2.add_artist(legend2)
ax2.set_title('t-SNE of Digits Dataset')

plt.show()
