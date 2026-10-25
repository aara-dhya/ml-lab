import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

funcs = {
    "Sigmoid": 1 / (1 + np.exp(-x)),
    "Tanh": np.tanh(x),
    "ReLU": np.maximum(0, x),
    "Leaky ReLU": np.where(x > 0, x, 0.1 * x)
}

for i, (name, y) in enumerate(funcs.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(x, y)
    plt.title(name); plt.grid(True)

plt.tight_layout(); plt.savefig("plot.png")