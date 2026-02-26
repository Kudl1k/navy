import numpy as np


class Perceptron:

    def __init__(self, value, l_r=0.1, n_iter=100):
        self.weights = None                                         # vektor
        self.iteration_count = None                                 # pocet iteraci, nez zkonverguje
        self.l_r = l_r                                              # learning rate
        self.n_iter = n_iter                                        # maximalni pocet iteraci
        self.set_weights(value)                                    # inicializace vektoru

    def set_weights(self, n_features):
        self.weights = np.random.randn(n_features.size + 1) * 0.01  # inicializace vektoru (+1 pro bias)

    @staticmethod
    def activate(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    def predict(self, X):
        z = np.dot(X, self.weights[1:]) + self.weights[0]          # linearni kombinace vstupu a vah
        return self.activate(z)                                    # aktivacni funkce

    def train(self, X, y):
        self.iteration_count = 0
        for _ in range(self.n_iter):
            predicted_all = True
            for xi, target in zip(X, y):
                prediction = self.predict(xi)                       # udelame predikci pro vzorek
                if prediction != target:                            # pokud se predikce neshoduje s targetem, musime aktualizovat vektory vah
                    predicted_all = False
                    update = self.l_r * (target - prediction)       # vypocet aktualizace vah
                    self.weights[1:] += update * xi                 # Aktualizace vah pro vstupni vektory
                    self.weights[0] += update                       # Aktualizace vah pro bias

            if predicted_all:
                break
            else:
                self.iteration_count += 1
