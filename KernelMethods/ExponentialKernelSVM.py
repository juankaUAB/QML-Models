import pennylane as qml
from pennylane import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_exponential_kernel(x1, x2):
    # Primero haremos el caso en el que no incluiremos una variable theta personalizada para las rotaciones. Más adelante lo probaremos.
    
    # Encode the input vectors x1 and x2
    qml.templates.AngleEmbedding(x1, wires=range(len(x1)))
    qml.templates.AngleEmbedding(x2, wires=range(len(x2)))

    return qml.expval(qml.PauliZ(0))

def exponential_kernel(x1, x2):
    return np.exp(-0.5 * quantum_exponential_kernel(x1, x2))

class QuantumExponentialKernel:
    def __init__(self):
        self.dev = qml.device("default.qubit", wires=4)
        
    def quantum_exponential_kernel(self, x, y):
        return quantum_exponential_kernel(x, y)

    def __call__(self, X, Y):
        kernel_matrix = np.zeros((len(X), len(Y)))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                kernel_matrix[i, j] = exponential_kernel(x, y)
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

q_exponential_kernel = QuantumExponentialKernel()

model = svm.SVC(kernel=q_exponential_kernel)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Results:", sum(predictions == y_test) / len(predictions))