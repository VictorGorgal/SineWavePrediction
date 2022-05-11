import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import generator
import numpy as np


def add(array, element):
    new = np.zeros(array.shape)
    new[0, :-1, 0] = array[0, 1:, 0]
    new[0, -1, 0] = element
    return new


x = np.linspace(0, 2, 200)
X = generator.generate2(x, amplitude=1, frequency=2)
y = X
X = X.reshape((1, len(x), 1))

model = load_model('model.h5')

y_hat = []
for i in range(X.shape[1]):
    pred = model.predict(X)[0, 0]

    y_hat.append(pred)
    X = add(X, pred)

# x = np.linspace(0, 2, 400)
# y = generator.generate2(x, frequency=1)
# y = y.reshape((len(y), 1))
#
# time_series = TimeseriesGenerator(y, y, length=200, sampling_rate=1, stride=1, batch_size=32)
#
# model = load_model('model.model')
# y_hat = model.predict(time_series)

x2 = np.linspace(1, 2, len(y_hat))
plt.plot(x, y, label='actual')
plt.plot(x2, y_hat, linestyle=':', label='predicted')
plt.legend()
plt.show()
