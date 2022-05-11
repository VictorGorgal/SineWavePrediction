import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import load_model
import matplotlib.pyplot as plt
import generator
import numpy as np


def add(array, element):
    new = np.zeros(array.shape)
    new[0, :-1, 0] = array[0, 1:, 0]
    new[0, -1, 0] = element
    return new


def predict(model, data, future=144):
    data = data.reshape(1, len(data), 1)
    y_hat = []
    for _ in range(future):
        pred = model.predict(data)[0, 0]

        y_hat.append(pred)
        data = add(data, pred)

    return y_hat


x = np.linspace(0, 2, 200)
y = generator.generate2(x, amplitude=1, frequency=2)

model = load_model('model.h5')

y_hat = predict(model, y, future=200)

x2 = np.linspace(1, 2, len(y_hat))
plt.plot(x, y.flatten(), label='actual')
plt.plot(x2, y_hat, linestyle=':', label='predicted')
plt.legend(loc='upper right')
plt.show()
