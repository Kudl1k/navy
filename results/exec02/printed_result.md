**1. Váhy a bias před trénováním**

| Vrstva / Neuron | Váhy / Bias |
| :--- | :--- |
| `neuron_hidden1.weights` | [0.792491, 0.464214] |
| `neuron_hidden2.weights` | [0.739202, 0.563267] |
| `neuron_output.weights` | [-0.066850, -0.353135] |
| `neuron_hidden1.bias` | 0.351069 |
| `neuron_hidden2.bias` | -0.764115 |
| `neuron_output.bias` | -0.483163 |

**2. Váhy a bias po trénování**

| Vrstva / Neuron | Váhy / Bias |
| :--- | :--- |
| `neuron_hidden1.weights` | [6.333507, 6.331653] |
| `neuron_hidden2.weights` | [4.717850, 4.717321] |
| `neuron_output.weights` | [9.617683, -10.218901] |
| `neuron_hidden1.bias` | -2.810035 |
| `neuron_hidden2.bias` | -7.238531 |
| `neuron_output.bias` | -4.485130 |

**3. Výsledky testování**

| Odhad sítě (Guess) | Očekávaný výstup | Shoduje se? |
| :--- | :---: | :---: |
| `0.0189590575` | **0** | True |
| `0.9836390862` | **1** | True |
| `0.9836410930` | **1** | True |
| `0.0168829279` | **0** | True |

Celková úspěšnost: **100.0 %**
