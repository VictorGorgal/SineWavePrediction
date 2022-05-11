# SineWavePrediction
Simple LSTM deep neural network to predict a sine wave function

generator.py - generates the sine wave

nn.py - used for training and saving the model to model.h5

prediction.py - is used to make new predictions using the saved model

the neural network is a encoder-decoder type, with 4 layers, 4 -> 1 -> 4 -> 1  neurons respectively.
![image](https://user-images.githubusercontent.com/94933775/167906581-8eee00eb-ba5e-4f3d-83e4-1450ead5c325.png)

prediction.py output:
![image](https://user-images.githubusercontent.com/94933775/167907365-2976812f-3639-4f25-b93b-d34ee0fe2c8e.png)
