import pennylane as qml
from pennylane import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def quantum_sigmoidal_kernel(x1, x2, theta):
    # Aqui meteremos la variable extra theta para ver si mejora el accuracy del modelo.
    
    # Encode the input vectors x1 and x2
    qml.templates.AngleEmbedding(x1, wires=range(len(x1)))
    qml.templates.AngleEmbedding(x2, wires=range(len(x2)))
    
    for i in range(len(x1)):
        qml.RY(theta[i], wires=i)

    return qml.expval(qml.Hadamard(0))

def sigmoidal_kernel(x1, x2, theta):
    # Rescale the angles
    scaled_theta = np.pi * theta
    
    return 1 / (1 + np.exp(-quantum_sigmoidal_kernel(x1, x2, scaled_theta)))

class QuantumSigmoidalKernel:
    def __init__(self, theta):
        self.dev = qml.device("default.qubit", wires=4)
        self.theta = theta
        
    def quantum_sigmoidal_kernel(self, x, y):
        return quantum_sigmoidal_kernel(x, y)

    def __call__(self, X, Y):
        kernel_matrix = np.zeros((len(X), len(Y)))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                kernel_matrix[i, j] = sigmoidal_kernel(x, y, self.theta)
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

theta = np.array([0.6, 0.5, 0.5, 0.4]) # Definimos los angulos theta que utilizaremos en el kernel

q_sigmoidal_kernel = QuantumSigmoidalKernel(theta)

model = svm.SVC(kernel=q_sigmoidal_kernel)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Results:", sum(predictions == y_test) / len(predictions))