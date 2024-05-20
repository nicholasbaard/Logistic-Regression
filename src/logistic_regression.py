import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def compute_cost(X:np.array, y:np.array, theta:np.array):
    m = y.size
    h = sigmoid(np.dot(X, theta))
    J = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return J

def compute_cost_softmax(X:np.array, y:np.array, theta:np.array):
    num_samples = X.shape[0]
    scores = np.dot(X, theta)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_samples), y])
    return np.sum(correct_logprobs) / num_samples

def logistic_regression(X:np.array, y:np.array, theta:np.array, alpha:float=0.01, eps:float=0.0001, lmbda:float=0.001):
    m, n = X.shape
    cost_history = []
    theta_old = np.zeros_like(theta)

    #loop until convergence:
    while np.sqrt(np.sum(np.power(theta - theta_old, 2))) > eps:
        theta_old = theta
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = (1 / m) * np.dot(X.T, (h - y)) + (lmbda * np.sum(np.power(theta, 2)))
        theta = theta - alpha * gradient

        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history
