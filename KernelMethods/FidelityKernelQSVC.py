from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 12345

X, y = load_iris(return_X_y=True)

# pick inputs and labels from the first two classes only,
# corresponding to the first 100 samples
X = X[:100]
y = y[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y)

dim = len(X_train[0])

iris_feature_map = ZZFeatureMap(feature_dimension=dim, reps=2, entanglement="linear")
sampler = Sampler()
iris_fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(feature_map=iris_feature_map, fidelity=iris_fidelity)

model = QSVC(quantum_kernel=quantum_kernel)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Results:", sum(predictions == y_test) / len(predictions))