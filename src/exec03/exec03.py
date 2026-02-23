import numpy as np


class HopfieldNetwork:
    def __init__(self):
        self.W = None

    @staticmethod
    def prepare_vector(data: np.ndarray):
        data_adjusted = np.where(data == 1, 1, -1)      # prevod z 1 a 0 na 1 a -1
        return data_adjusted.flatten()                    # prevod 2D matice na 1D vektor

    def phase_learning(self, patterns: list[np.ndarray]):
        num_neurons = len(patterns[0]) # velikost vektoru

        self.W = np.zeros((num_neurons, num_neurons))   # inicializace matice plnou nulami

        for pattern in patterns:

            W_pattern = np.outer(pattern, pattern)      # vahova matice pomoci X . X^T

            self.W += W_pattern                         # aktualizace vah

        np.fill_diagonal(self.W, 0)                 # nastaveni diagonalky na nulu


    def sync_recovery(self, corrupted_vector: np.ndarray):
        raw_result = np.dot(self.W, corrupted_vector)           # vynasobeni vah a vektoru
        recovered_vector = np.where(raw_result >= 0, 1, -1)  # prevod na 1 a -1 (0 je prahova hodnota)
        return recovered_vector

    def async_recovery(self, corrupted_vector: np.ndarray, max_iterations: int = 1000, callback=None):
        recovered_vector = corrupted_vector.copy()              # kopie vektoru, ktery budeme postupne opravovat
        num_neurons = len(recovered_vector)                     # velikost vektoru

        for iteration in range(max_iterations):                             # pocet iteraci pro maximalni opravu vektoru
            previous_vector = recovered_vector.copy()                       # kopie vektoru pro porovnavani

            for i in range(num_neurons):                                    # prochazeni postupne vsech neuronů a aktualizace jejich hodnot
                raw_result = np.dot(self.W[:, i], recovered_vector)         # vynásobení vstupního vektoru a jednoho sloupce váhové matice

                new_value = 1 if raw_result >= 0 else -1                    # prevod na 1 a -1 (0 je prahova hodnota)

                if recovered_vector[i] != new_value:                        # aktualizace pouze pokud se hodnota zmeni
                    recovered_vector[i] = new_value                         # aktualizace hodnoty

                    if callback:                                            # volani callback funkce pro tkinter aktualizace canvasu
                        callback(recovered_vector)

            if np.array_equal(previous_vector, recovered_vector):
                print(f"Recovery converged after {iteration + 1} iterations.")
                break

        return recovered_vector