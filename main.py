import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Assuming the last column is the target column
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def min_max_scaling(X):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return (X - min_vals) / (max_vals - min_vals)

X_train = min_max_scaling(X_train)
X_test = min_max_scaling(X_test)

# Implementing logistic regression algorithm
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
        cost = cost_function(theta, X, y)
        cost_history.append(cost)
    return theta, cost_history

num_iterations = 100
learning_rate = 0.1

X_train = np.c_[np.ones((len(X_train), 1)), X_train]
X_test = np.c_[np.ones((len(X_test), 1)), X_test]

# Initializing theta with zeros
theta = np.zeros(X_train.shape[1])

# Performing gradient descent to obtain the optimized theta and cost history
opt_theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

# Plotting the graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_iterations + 1), cost_history, color='blue', linewidth=2)
plt.xlabel('Number of Iterations', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title('Cost vs Number of Iterations', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.show()


