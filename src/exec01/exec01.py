import numpy as np


class Perceptron:

    def __init__(self, value, l_r=0.1, n_iter=100):
        self.iteration_count = None                                 # Number of iterations until convergence
        self.l_r = l_r                                              # Learning rate
        self.n_iter = n_iter                                        # Maximum number of iterations (Epochs)
        self._set_weights(value)                                    # Initialize weights

    def _set_weights(self, n_features):
        self.weights = np.random.randn(n_features.size + 1) * 0.01  # Initialize weights to small random values

    @staticmethod
    def _activate(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    def _predict(self, X):
        z = np.dot(X, self.weights[1:]) + self.weights[0]           # Linear combination of inputs and weights
        return self._activate(z)                                    # Activation function

    def train(self, X, y):
        self.iteration_count = 0
        for _ in range(self.n_iter):
            predicted_all = True
            for xi, target in zip(X, y):
                prediction = self._predict(xi)                      # Make a prediction
                if prediction != target:                            # Check if it was wrong
                    predicted_all = False
                    update = self.l_r * (target - prediction)       # Calculate weight update
                    self.weights[1:] += update * xi                 # Update weights
                    self.weights[0] += update                       # Also update bias weight

            if predicted_all:
                break
            else:
                self.iteration_count += 1
