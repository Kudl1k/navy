import numpy as np
import os
import matplotlib.pyplot as plt

from src.help import save_figure


class XORNetwork:
    def __init__(self):
        self.weights_H = np.random.randn(2, 2)          # 2 vstupy -> 2 skryté neurony
        self.bias_H = np.random.randn(1, 2)             # bias pro 2 skryté neurony

        self.weights_O = np.random.randn(2, 1)          # 2 skryté -> 1 výstupní neuron
        self.bias_O = np.random.randn(1, 1)             # bias pro výstupní neuron

    def print_weights(self):
        print(f"{'neuron_hidden1.weights':<25}[{self.weights_H[0, 0]}, {self.weights_H[1, 0]}]")
        print(f"{'neuron_hidden2.weights':<25}[{self.weights_H[0, 1]}, {self.weights_H[1, 1]}]")
        print(f"{'neuron_output.weights':<25}[{self.weights_O[0, 0]}, {self.weights_O[1, 0]}]")
        print(f"{'neuron_hidden1.bias':<25}{self.bias_H[0, 0]}")
        print(f"{'neuron_hidden2.bias':<25}{self.bias_H[0, 1]}")
        print(f"{'neuron_output.bias':<25}{self.bias_O[0, 0]}")

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))                     # Sigmoid - aktivacni funkce

    @staticmethod
    def sigmoid_(out):
        return out * (1 - out)                          # Derivace sigmoidu

    def predict(self, inputs):
        out_H = self.sigmoid(np.dot(inputs, self.weights_H) + self.bias_H)  # Vypocet vystupu pro skryte neurony
        out_O = self.sigmoid(np.dot(out_H, self.weights_O) + self.bias_O)   # Vypocet vystupu pro vystupni neuron
        return out_O, out_H

    def train(self, inputs, targets, iterations=10000, learning_rate=0.5):
        loss_history = [] # pro sledování vývoje chyby během trénování
        for i in range(iterations):

            # Feedforward
            out_O, out_H = self.predict(inputs)

            # Vypocet celkove chyby
            error = targets - out_O

            if i % 100 == 0:
                mse = np.mean(error ** 2)
                loss_history.append(mse)

            # Backpropagation
            delta_O = error * self.sigmoid_(out_O)                              # chyba na vystupu pomoci derivace sigmoidu
            delta_H = np.dot(delta_O, self.weights_O.T) * self.sigmoid_(out_H)  # prenasime chybu z vystupu do skrytych neuronu

            # Aktualizace vah a biasů | Gradient Descent
            self.weights_O += learning_rate * np.dot(out_H.T, delta_O)
            self.bias_O += learning_rate * np.sum(delta_O, axis=0, keepdims=True)

            self.weights_H += learning_rate * np.dot(inputs.T, delta_H)
            self.bias_H += learning_rate * np.sum(delta_H, axis=0, keepdims=True)

        return loss_history

def visualize_results(loss_history, nn, inputs, targets):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(0, len(loss_history) * 100, 100), loss_history, color='blue', lw=2)
    ax1.set_title('Křivka učení (Vývoj chyby)')
    ax1.set_xlabel('Iterace (Epochs)')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 100), np.linspace(-0.2, 1.2, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    predictions, _ = nn.predict(grid_points)
    predictions = predictions.reshape(xx.shape)

    contour = ax2.contourf(xx, yy, predictions, levels=20, cmap='RdBu', alpha=0.8)
    plt.colorbar(contour, ax=ax2, label='Predikce sítě (0 = Červená, 1 = Modrá)')

    ax2.scatter(inputs[:, 0], inputs[:, 1], c=targets.flatten(),
                cmap='bwr', edgecolors='white', s=150, zorder=3, linewidths=2)

    ax2.set_title('Rozhodovací prostor sítě pro XOR')
    ax2.set_xlabel('Vstup x1')
    ax2.set_ylabel('Vstup x2')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])

    plt.tight_layout()
    save_figure("exec02","xor_neural_network_result", plt)
    plt.show()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'results', 'exec02'))
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'printed_result.md')

    def generate_weights_md_table(network):
        return (
            "| Vrstva / Neuron | Váhy / Bias |\n"
            "| :--- | :--- |\n"
            f"| `neuron_hidden1.weights` | [{network.weights_H[0, 0]:.6f}, {network.weights_H[1, 0]:.6f}] |\n"
            f"| `neuron_hidden2.weights` | [{network.weights_H[0, 1]:.6f}, {network.weights_H[1, 1]:.6f}] |\n"
            f"| `neuron_output.weights` | [{network.weights_O[0, 0]:.6f}, {network.weights_O[1, 0]:.6f}] |\n"
            f"| `neuron_hidden1.bias` | {network.bias_H[0, 0]:.6f} |\n"
            f"| `neuron_hidden2.bias` | {network.bias_H[0, 1]:.6f} |\n"
            f"| `neuron_output.bias` | {network.bias_O[0, 0]:.6f} |\n"
        )

    md_content = ""

    nn = XORNetwork()
    print("----------------------------------------------")
    print("Weight and bias before the learning phase:")
    nn.print_weights()

    md_content += "**1. Váhy a bias před trénováním**\n\n"
    md_content += generate_weights_md_table(nn) + "\n"

    print("----------------------------------------------")
    print("Learning in progress..")
    print("----------------------------------------------")

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    loss_history = nn.train(inputs, targets, iterations=10000)

    print("Weight and bias after the learning phase:")
    nn.print_weights()

    md_content += "**2. Váhy a bias po trénování**\n\n"
    md_content += generate_weights_md_table(nn) + "\n"

    print("----------------------------------------------")
    print("Testing in progress..")
    print("----------------------------------------------")

    print(f"{'Guess':<25}{'Expected output':<23}Is it equal?")

    md_content += "**3. Výsledky testování**\n\n"
    md_content += "| Odhad sítě (Guess) | Očekávaný výstup | Shoduje se? |\n"
    md_content += "| :--- | :---: | :---: |\n"

    correct_predictions = 0
    for x, target in zip(inputs, targets):
        out_O, _ = nn.predict(x.reshape(1, -1))
        guess_val = out_O[0, 0]
        expected = target[0]

        is_equal = (1 if guess_val > 0.5 else 0) == expected
        if is_equal:
            correct_predictions += 1

        print(f"{guess_val:<25}{expected:<23}{is_equal}")

        md_is_equal = "True" if is_equal else "False"
        md_content += f"| `{guess_val:.10f}` | **{expected}** | {md_is_equal} |\n"

    success_rate = (correct_predictions / len(inputs)) * 100
    print(f"success is {success_rate} %")
    print("----------------------------------------------")

    md_content += f"\nCelková úspěšnost: **{success_rate} %**\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Report byl úspěšně vygenerován a uložen do:\n> {file_path}")

    visualize_results(loss_history, nn, inputs, targets)


if __name__ == "__main__":
    main()