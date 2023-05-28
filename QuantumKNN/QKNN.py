import pennylane as qml
from pennylane import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

NUM_WIRES = 5

# Probando a disminuir la separación entre las dos clases, los datos se mezclan y se produce una peor clasificación (lo esperado)
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=2.0,
)

# Plot dataset
for data, label in zip(X, y):
    if label == 0:
        plt.scatter(data[0], data[1], color="orange")
    else:
        plt.scatter(data[0], data[1], color="blue")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=75, test_size=25)

plt.show()

def feature_map(x, wires):
    qml.AngleEmbedding(x, wires=wires, rotation="Y")
    
# |<x1|x2>|^2

dev = qml.device("default.qubit", wires = 5)

@qml.qnode(dev)
def swap_test(x1,x2):
    
    feature_map(x1, wires = [1,2])
    feature_map(x2, wires = [3,4])
    
    qml.Hadamard(wires = 0)
    qml.CSWAP(wires = [0,1,3])
    qml.CSWAP(wires = [0,2,4])
    qml.Hadamard(wires = 0)
    
    return qml.expval(qml.PauliZ(0))
    
def distance(x1, x2):
    return 2 - 2 * swap_test(x1,x2)

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3,metric=distance)

neigh.fit(X_train,y_train)
print("Accuracy: ", str(str(neigh.score(X_test, y_test) * 100) + "%"))