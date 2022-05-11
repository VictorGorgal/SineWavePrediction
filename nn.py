import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import generator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator


EPOCHS = 50

look_back = 200

X_train = np.linspace(0, 10, 2000)
y_train = generator.generate2(X_train, frequency=2)

X_test = np.linspace(10, 20, 2000)
y_test = generator.generate2(X_test, frequency=2)

train_series = y_train.reshape((len(y_train), 1))
test_series = y_test.reshape((len(y_test), 1))

train_generator = TimeseriesGenerator(train_series, train_series,
                                      length=look_back,
                                      sampling_rate=1,
                                      stride=1,
                                      batch_size=64)

test_generator = TimeseriesGenerator(test_series, test_series,
                                     length=look_back,
                                     sampling_rate=1,
                                     stride=1,
                                     batch_size=64)

neurons = 4

model = Sequential()
model.add(LSTM(neurons, input_shape=(look_back, 1), return_sequences=True))
model.add(LSTM(1, input_shape=(look_back,), return_sequences=True))
model.add(LSTM(neurons, input_shape=(1,)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

history = model.fit(train_generator, epochs=EPOCHS)

model.save('model.h5')

plt.plot(history.history['loss'])
plt.title('model loss')
plt.show()
