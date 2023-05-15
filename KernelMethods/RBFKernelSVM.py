import pennylane as qml
from pennylane import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_circuit(x):
    for i in range(len(x)):
        qml.Hadamard(wires=i)
        qml.RY(x[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(len(x))]

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(np.array(x1) - np.array(x2)) ** 2)

class QuantumRBFKernel:
    def __init__(self, gamma=0.5):
        self.dev = qml.device("default.qubit", wires=4)
        self.gamma = gamma
        
    def quantum_circuit(self, x):
        return quantum_circuit(x)

    def __call__(self, X, Y):
        kernel_matrix = np.zeros((len(X), len(Y)))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                feature_map_x = self.quantum_circuit(x)
                feature_map_y = self.quantum_circuit(y)
                kernel_matrix[i, j] = rbf_kernel(feature_map_x, feature_map_y, self.gamma)
        return kernel_matrix


X, y = load_iris(return_X_y=True)

# pick inputs and labels from the first two classes only,
# corresponding to the first 100 samples
X = X[:100]
y = y[:100]

# scaling the inputs is important since the embedding we use is periodic
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# scaling the labels to -1, 1 is important for the SVM and the
# definition of a hinge loss
y_scaled = 2 * (y - 0.5)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

gamma = 0.5

quantum_linear_kernel = QuantumRBFKernel(gamma)

model = svm.SVC(kernel=quantum_linear_kernel)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Results:", sum(predictions == y_test) / len(predictions))