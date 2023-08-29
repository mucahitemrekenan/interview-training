"""
Task: Central Limit Theorem Simulation

Objective:
Demonstrate the Central Limit Theorem by simulating the sampling distribution of the mean for a non-normally distributed population.

Steps:

Generate a non-normally distributed population of size N. For simplicity, you can use a uniform distribution or an exponential distribution.
Randomly draw n samples from the population and compute their mean. Repeat this process k times to simulate k sample means.
Plot a histogram of the k sample means.
Observe whether the distribution of the sample means approaches a normal distribution as k increases.
Requirements:

Use a population size N of 10,000.
For each simulation, draw a sample of size n=50.
Start with k=100 simulations and gradually increase to k=10,000.
Plot the histograms for different values of k to observe the effect.
"""


import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000  # Population size
n = 50     # Sample size

# Generate a non-normally distributed population (e.g., exponential distribution)
population = np.random.exponential(scale=1, size=N)


# Function to simulate sample means
def simulate_sample_means(k):
    means = []
    for _ in range(k):
        sample = np.random.choice(population, n)
        means.append(np.mean(sample))
    return means


# Simulate and plot
for k in [100]:
    sample_means = simulate_sample_means(k)
    plt.hist(sample_means, bins=50, density=True, alpha=0.6, label=f'k={k}')

plt.title('Sampling Distribution of the Mean')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.legend()
plt.show()

# Bonus: Compute and print mean and standard deviation
mean_of_sample_means = np.mean(sample_means)
std_of_sample_means = np.std(sample_means)
print(f'Mean of sample means: {mean_of_sample_means}')
print(f'Standard deviation of sample means: {std_of_sample_means}')

