import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from NNModel import NeuralNetwork
from NPModel import NPNeuralNetwork
from Stats import process_results
import pandas as pd


X_train, _, y_train, _, _ = load_data()

epochs = 400
architecture = [784, 32, 32, 10]

print("Training standard neural network...")
std_model = NeuralNetwork(architecture)
std_model.train(X_train, y_train, epochs=epochs)
std_model.save_model()

print("\nTraining neuroplastic neural network...")
np_model = NPNeuralNetwork(architecture)
np_model.train(X_train, y_train, epochs=epochs)
np_model.save_model()

process_results(std_model, np_model)