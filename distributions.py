"""
A Gaussian distribution, also known as a normal distribution,
is a type of continuous probability distribution for a real-valued random variable.
It is characterized by two parameters: the mean (μ) and the standard deviation (σ).
The Gaussian distribution is symmetric and has the familiar "bell curve" shape.

A non-Gaussian distribution is any distribution that does not have the properties of a Gaussian distribution.
There are many types of non-Gaussian distributions, such as the uniform distribution, exponential distribution,
binomial distribution, and many others.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate random numbers from a Gaussian distribution
mu, sigma = 0, 0.1  # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

# Plot histogram
plt.hist(s, bins=30, density=True)
plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# Generate random numbers from an exponential distribution
scale = 2.0  # mean and inverse of the rate parameter (lambda)
s_exp = np.random.exponential(scale, 1000)

# Plot histogram
plt.hist(s_exp, bins=30, density=True)
plt.title('Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()