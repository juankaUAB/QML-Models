import numpy as np

from qiskit import BasicAer
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Using difficult dataset (the two classes are not easily differentiated) but easy for a quantum feature map

# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)

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
vqc = VQC(NUM_WIRES, feature_map, ansatz, optimizer = optimizer, callback=callback_graph)
vqc.fit(X_train, y_train)
print(vqc.fit_result)

score = vqc.score(X_test, y_test)
print("Final accuracy: ", score)

plot_track(objective_func_vals, 30)



