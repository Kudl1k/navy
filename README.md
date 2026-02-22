# NAVY
## Task 1: Percepton - point on the line
Generate the set of 100 points and use a perceptron to find out whether the
points lie above, below, or on the line.
![Perceptron](/results/exec01/perceptron_chart_lr0.10_iter100_samples100.png)
## Task 2: Simple neural network: XOR problem
Use a neural network to solve the XOR problem
**1. Váhy a bias před trénováním**

| Vrstva / Neuron | Váhy / Bias |
| :--- | :--- |
| `neuron_hidden1.weights` | [0.012174, 1.704211] |
| `neuron_hidden2.weights` | [-1.182833, -0.482396] |
| `neuron_output.weights` | [0.574572, 1.020012] |
| `neuron_hidden1.bias` | -0.466490 |
| `neuron_hidden2.bias` | -0.562764 |
| `neuron_output.bias` | 1.284588 |

**2. Váhy a bias po trénování**

| Vrstva / Neuron | Váhy / Bias |
| :--- | :--- |
| `neuron_hidden1.weights` | [-5.825349, 6.063918] |
| `neuron_hidden2.weights` | [-6.078899, 6.019559] |
| `neuron_output.weights` | [-9.272610, 9.581321] |
| `neuron_hidden1.bias` | 2.911744 |
| `neuron_hidden2.bias` | -3.284761 |
| `neuron_output.bias` | 4.394212 |

**3. Výsledky testování**

| Odhad sítě (Guess) | Očekávaný výstup | Shoduje se? |
| :--- | :---: | :---: |
| `0.0170519824` | **0** | True |
| `0.9840167995` | **1** | True |
| `0.9804996937` | **1** | True |
| `0.0152040544` | **0** | True |

![Neural network](/results/exec02/xor_neural_network_result.png)

Celková úspěšnost: **100.0 %**



