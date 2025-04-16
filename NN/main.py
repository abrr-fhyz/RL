import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from NNModel import NeuralNetwork
from NPModel import NPNeuralNetwork

def show_comparison_stats(acc_1, acc_2, lss_1, lss_2, label_1='Standard NN', label_2='NP NN'):
    epochs_1 = range(1, len(acc_1) + 1)
    epochs_2 = range(1, len(acc_2) + 1)
    acc_1_percent = [a * 100 for a in acc_1]
    acc_2_percent = [a * 100 for a in acc_2]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(epochs_1, acc_1_percent, label=f'{label_1} Accuracy', color='blue')
    ax1.plot(epochs_2, acc_2_percent, label=f'{label_2} Accuracy', color='green')
    ax1.plot(epochs_1[-1], acc_1_percent[-1], 'o', color='blue')
    ax1.plot(epochs_2[-1], acc_2_percent[-1], 'o', color='green')
    ax1.annotate(f'{acc_1_percent[-1]:.2f}%', (epochs_1[-1], acc_1_percent[-1]),
                 textcoords="offset points", xytext=(-30,10), ha='center',
                 fontsize=8, color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue'))
    ax1.annotate(f'{acc_2_percent[-1]:.2f}%', (epochs_2[-1], acc_2_percent[-1]),
                 textcoords="offset points", xytext=(30,10), ha='center',
                 fontsize=8, color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.set_title("Model Comparison: Accuracy")

    ax2.plot(epochs_1, lss_1, label=f'{label_1} Loss', color='red')
    ax2.plot(epochs_2, lss_2, label=f'{label_2} Loss', color='orange')
    ax2.plot(epochs_1[-1], lss_1[-1], 'o', color='red')
    ax2.plot(epochs_2[-1], lss_2[-1], 'o', color='orange')
    ax2.annotate(f'{lss_1[-1]:.5f}', (epochs_1[-1], lss_1[-1]),
                 textcoords="offset points", xytext=(-30,-10), ha='center',
                 fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate(f'{lss_2[-1]:.5f}', (epochs_2[-1], lss_2[-1]),
                 textcoords="offset points", xytext=(30,-10), ha='center',
                 fontsize=8, color='orange',
                 arrowprops=dict(arrowstyle='->', color='orange'))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_title("Model Comparison: Loss")

    plt.tight_layout()
    plt.show()

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, X_test, y_train, y_test

def predict(model, X_test, y_test):
    preds = model.predict(X_test)
    predicted_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

X_train, X_test, y_train, y_test = load_data()

epochs = 500
architecture = [784, 16, 16, 10]

model_1 = NeuralNetwork(architecture)
model_1.train(X_train, y_train, epochs=epochs)
model_1.save_model()

model_2 = NPNeuralNetwork(architecture)
model_2.train(X_train, y_train, epochs=epochs)
model_2.save_model()

acc_1, lss_1 = model_1.get_stats()
acc_2, lss_2 = model_2.get_stats()

predict(model_1, X_test, y_test)
predict(model_2, X_test, y_test)

show_comparison_stats(acc_1, acc_2, lss_1, lss_2)





