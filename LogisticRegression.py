import numpy as np
import math


class LogisticRegression:

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights = None

    def train(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)

        cost_list = []

        for _ in range(self.n_iterations):
            linear_model = np.dot(x, self.weights)
            y_predicted = self._sigmoid(linear_model)

            dw = 0

            for _ in range(n_samples):
                dw += (np.dot(x.T, y_predicted - y)) / (1 + np.exp(np.dot(x.T, y_predicted - y)))
            dw = 1 / n_samples * dw

            self.weights -= self.learning_rate * dw

            cost = -(1 / n_samples) * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))

            cost_list.append(cost)

        return cost_list

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
