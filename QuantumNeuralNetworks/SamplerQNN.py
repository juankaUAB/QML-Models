import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.model_selection import train_test_split

algorithm_globals.random_seed = 42

def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)

def plot_track(values, EPOCHS):
    func_values = values
    epochs = range(1,EPOCHS+1)
    plt.plot(epochs, func_values, 'g')
    plt.title('Training metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Objective function value')
    plt.legend()
    plt.show()

NUM_WIRES = 2

X, y, _, _ = ad_hoc_data(100, 50, NUM_WIRES, 0.3)
y = np.array([np.argwhere(i)[0,0] for i in y])

# plot dataset
for x, y_target in zip(X, y):
    if y_target == 1:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "go")
plt.plot()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150, test_size=50)

qc = QuantumCircuit(NUM_WIRES)
feature_map = ZZFeatureMap(NUM_WIRES, reps=3)
ansatz = TwoLocal(NUM_WIRES, ['ry', 'rz'], 'cz', reps=3)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)
qc.draw(output="mpl", interactive=True)
plt.show()

# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2


output_shape = 2  # corresponds to the number of classes, possible outcomes of the (parity) mapping.

sampler_qnn = SamplerQNN(circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=output_shape,)

sampler_classifier = NeuralNetworkClassifier(
    neural_network=sampler_qnn, optimizer=COBYLA(maxiter=30), callback=callback_graph
) # loss function predetermined (test with other loss functions)

objective_func_vals = []
sampler_classifier.fit(X_train, y_train)

accuracy = sampler_classifier.score(X_test, y_test)
print("Final accuracy: ", accuracy)
plot_track(objective_func_vals, 30)