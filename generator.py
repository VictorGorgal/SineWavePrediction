import numpy as np
import matplotlib.pyplot as plt


def generate(amplitude=1, frequency=1):
    x = np.linspace(0, 50, 100)
    y = amplitude * np.sin(2 * np.pi * frequency * x)

    return x, y


def generate2(x, amplitude=1, frequency=1):
    y = amplitude * np.sin(2 * np.pi * frequency * x)

    return y


if __name__ == '__main__':
    x, y = generate(amplitude=1, frequency=2)

    plt.plot(x, y)
    plt.show()
