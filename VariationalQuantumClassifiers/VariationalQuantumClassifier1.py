import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Simple VQC, using Nesterov Momentum Optimizer (we can test it with others), with simple data embedding (angle embedding), simple cost function (square loss), and simple ansatz. (very good results! 😛)

np.random.seed(0)

dev = qml.device("default.qubit", wires=4)

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def statepreparation(x):
    qml.AngleEmbedding(x, wires=range(4))

def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    
@qml.qnode(dev, interface="autograd")
def circuit(weights, features):
    statepreparation(features)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, features):
    return circuit(weights, features) + bias


def cost(weights, bias, features, labels):
    predictions = [variational_classifier(weights, bias, f) for f in features]
    return square_loss(labels, predictions)

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

X, y = load_iris(return_X_y=True)

# pick inputs and labels from the first two classes only,
# corresponding to the first 100 samples
X = X[:100]
y = y[:100]

# shift label from {0, 1} to {-1, 1}
y = y * 2 - np.ones(len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y)

num_qubits = 4
num_layers = 6

weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

opt = NesterovMomentumOptimizer(0.01)
batch_size = 10

# train the variational classifier
weights = weights_init
bias = bias_init
epoch_cost = []
epoch_accuracy = []
for it in range(30):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    feats_train_batch = X_train[batch_index]
    Y_train_batch = y_train[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, feats_train_batch, Y_train_batch)

    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier(weights, bias, f)) for f in X_train]
    predictions_test = [np.sign(variational_classifier(weights, bias, f)) for f in X_test]

    # Compute accuracy on train and validation set
    acc_train = accuracy(y_train, predictions_train)
    acc_val = accuracy(y_test, predictions_test)
    
    cost_value = cost(weights, bias, X, y)
    epoch_cost.append(cost_value)
    epoch_accuracy.append(acc_train)

    print(
        "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc test: {:0.7f} "
        "".format(it + 1, cost_value, acc_train, acc_val)
    )

def plot_data(Diag_Cost, Diag_Acc, EPOCHS):
    loss_train = Diag_Cost
    acc_train = Diag_Acc
    epochs = range(1,EPOCHS+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, acc_train, 'r', label='Training Accuracy')
    plt.title('Training metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Accuracy')
    plt.legend()
    plt.show()
    
plot_data(epoch_cost, epoch_accuracy, 30)