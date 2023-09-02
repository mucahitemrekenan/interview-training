"""
Task: Sampling Techniques Demonstration

Objective:
Implement and compare various sampling techniques on a dataset.

Steps:

Dataset Generation: Create a dataset of 10,000 data points with two features:
age (ranging from 1 to 100) and income (ranging from $1,000 to $100,000).
Assume a certain correlation between age and income for added complexity.

Simple Random Sampling: Randomly select 1,000 data points from the dataset without a replacement.

Stratified Sampling: Divide the dataset into strata based on age groups (e.g., 1-20, 21-40, 41-60, 61-80, 81-100).
From each stratum, randomly select a proportionate number of samples such that the total is 1,000.

Cluster Sampling: Divide the dataset into 100 clusters.
Randomly select 10 clusters and take all the data points from these selected clusters.

Systematic Sampling: Starting from a random point, select every 10th data point until you have 1,000 samples.

Visualization: Plot the original dataset and the samples from each method on separate scatter plots.
Use different colors for different sampling methods.

Analysis: Compute the mean and standard deviation of the 'income' feature for the original dataset
and for each of the sampled datasets.
Compare how close the statistics of the sampled datasets are to the original dataset.

Requirements:

Use Python and libraries like NumPy and Matplotlib (or any other relevant libraries).
Ensure that the generated dataset has a discernible pattern or correlation between age and income.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate dataset
np.random.seed(42)
ages = np.random.randint(1, 101, 10000)
incomes = np.random.randint(1000, 100001, 10000) + ages * 500

# Simple Random Sampling
srs_sample = np.random.choice(np.arange(10000), 1000, replace=False)

# Stratified Sampling
strata_indices = [np.where((ages >= i) & (ages < i + 20))[0] for i in range(1, 101, 20)]
stratified_sample = np.concatenate([np.random.choice(stratum, 200, replace=False) for stratum in strata_indices])

# Cluster Sampling
clusters = np.array_split(np.arange(10000), 100)
selected_cluster_indices = np.random.choice(len(clusters), 10, replace=False)
cluster_sample = np.concatenate([clusters[idx] for idx in selected_cluster_indices])

# Systematic Sampling
start = np.random.randint(0, 10)
systematic_sample = np.arange(start, 10000, 10)[:1000]

# Visualization
plt.scatter(ages, incomes, label='Original Data', alpha=0.2)
plt.scatter(ages[srs_sample], incomes[srs_sample], label='SRS', alpha=0.5)
plt.scatter(ages[stratified_sample], incomes[stratified_sample], label='Stratified', alpha=0.5)
plt.scatter(ages[cluster_sample], incomes[cluster_sample], label='Cluster', alpha=0.5)
plt.scatter(ages[systematic_sample], incomes[systematic_sample], label='Systematic', alpha=0.5)
plt.legend()
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Sampling Techniques Demonstration')
plt.show()

# Analysis
methods = ['Original', 'SRS', 'Stratified', 'Cluster', 'Systematic']
samples = [np.arange(10000), srs_sample, stratified_sample, cluster_sample, systematic_sample]
for method, sample in zip(methods, samples):
    print(f'{method} - Mean Income: {np.mean(incomes[sample]):.2f}, Std Dev: {np.std(incomes[sample]):.2f}')

# Bonus: Convenience Sampling
convenience_sample = np.arange(1000)
print(f'Convenience - Mean Income: {np.mean(incomes[convenience_sample]):.2f}, '
      f'Std Dev: {np.std(incomes[convenience_sample]):.2f}')
