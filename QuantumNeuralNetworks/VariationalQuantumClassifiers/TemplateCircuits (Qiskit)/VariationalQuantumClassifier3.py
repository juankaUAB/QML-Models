import numpy as np
from qiskit import BasicAer
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 42

# Using difficult dataset (multi class, instead of binary) to take advantage of VQC algorithm

# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)

NUM_WIRES = 2

X, y = make_classification(
    n_samples=200,
    n_features=NUM_WIRES,
    n_classes=3,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=algorithm_globals.random_seed,
)
X = MinMaxScaler().fit_transform(X)

# plot dataset
for x, y_target in zip(X, y):
    if y_target == 0:
        plt.plot(x[0], x[1], "bo")
    elif y_target == 1:
        plt.plot(x[0], x[1], "go")
    else:
        plt.plot(x[0], x[1], "ro")
plt.plot()
plt.show()

# plot track of tranining step
def plot_track(values, EPOCHS):
    func_values = values
    epochs = range(1,EPOCHS+1)
    plt.plot(epochs, func_values, 'g')
    plt.title('Training metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Objective function value')
    plt.legend()
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150, test_size=50)

ansatz = TwoLocal(NUM_WIRES, ['ry', 'rz'], 'cz', reps=3)
feature_map = ZZFeatureMap(feature_dimension=NUM_WIRES, reps=3)
optimizer = COBYLA(maxiter=30)

objective_func_vals = []
vqc = VQC(NUM_WIRES, feature_map, ansatz, optimizer = optimizer, callback=callback_graph) # loss function predetermined (test with other loss functions)
vqc.fit(X_train, y_train)
print(vqc.fit_result)

score = vqc.score(X_test, y_test)
print("Final accuracy: ", score)

plot_track(objective_func_vals, 30)



