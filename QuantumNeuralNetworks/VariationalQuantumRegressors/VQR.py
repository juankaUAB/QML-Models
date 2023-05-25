import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.regressors import VQR
from sklearn.model_selection import train_test_split
from IPython.display import clear_output

algorithm_globals.random_seed = 42

def plot_track(values, EPOCHS):
    func_values = values
    epochs = range(1,EPOCHS+1)
    plt.plot(epochs, func_values, 'g')
    plt.title('Training metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Objective function value')
    plt.legend()
    plt.show()

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

num_samples = 200
eps = 0.2
lb, ub = -np.pi, np.pi
X_ = np.linspace(lb, ub, num=50).reshape(50, 1)
f = lambda x: np.sin(x)

X = (ub - lb) * algorithm_globals.random.random([num_samples, 1]) + lb
y = f(X[:, 0]) + eps * (2 * algorithm_globals.random.random(num_samples) - 1)

plt.plot(X_, f(X_), "r--")
plt.plot(X, y, "bo")
plt.show()


# construct simple feature map
param_x = Parameter("x")
feature_map = QuantumCircuit(1, name="fm")
feature_map.ry(param_x, 0)

# construct simple ansatz
param_y = Parameter("y")
ansatz = QuantumCircuit(1, name="vf")
ansatz.ry(param_y, 0)

# construct a circuit
qc = QuantumCircuit(1)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=150, test_size=50)

vqr = VQR(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=L_BFGS_B(maxiter=5),
    callback=callback_graph,
)

objective_func_vals = []
vqr.fit(X_train, y_train)

mean_square_error = vqr.score(X_test, y_test)
print("Final MSE: ", mean_square_error)

# plot target function
plt.plot(X_, f(X_), "r--")

# plot data
plt.plot(X, y, "bo")

# plot fitted line
y_ = vqr.predict(X_)
plt.plot(X_, y_, "g-")
plt.show()