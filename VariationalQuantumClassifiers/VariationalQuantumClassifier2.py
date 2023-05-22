import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# VQE 2, with Adam Optimizer (we can change it), more complex cost function, more complex data embedding, and PCA reduction to try to use less qubits.

NUM_WIRES = 3 # number of data features
NUM_LAYERS = 2 # number of repeated layers over quantum circuit

X, y = load_iris(return_X_y=True)

X = X[:150]

pca = PCA(n_components=3)
X = pca.fit_transform(X)

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# shift label from {0, 1, 2} to {-1, 0, 1}
y = y - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100, test_size=50)

dev = qml.device("default.qubit", wires=NUM_WIRES)

@qml.qnode(dev, interface="autograd")
def circuit(params,x):
    xEmbeded=[i*np.pi for i in x]
    for i in range(NUM_WIRES):
        qml.RX(xEmbeded[i],wires=i)
        qml.Rot(*params[0,i],wires=i)
    
    qml.CZ(wires=[1, 0])
    qml.CZ(wires=[1, 2])
    qml.CZ(wires=[0, 2])

    for i in range(NUM_WIRES):
        qml.Rot(*params[1,i],wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_WIRES)]

def prediction(weigths,x_train):
        predictions = [circuit(weigths, f) for f in x_train]
        for i,p in enumerate(predictions):
            predictions[i]=np.argmax(p)
        
        return predictions
    
def cost(weights, x_train, labels):
        predictions = [circuit(weights, f) for f in x_train]
        
        loss=0
        for i in range(len(predictions)):
            min=np.min(predictions[i])
            max=np.max(predictions[i])                   

            x=(predictions[i][labels[i]+1]-min)/(max-min)

            loss+= (1-x)**2 

        return loss/len(predictions)
    
def accuracy(weights,x_train,labels):
        predictions=prediction(weights,x_train)
        loss=0
        for i in range(len(predictions)):
            if predictions[i]==labels[i]:
                loss+=1
        loss=loss/len(predictions)
        return loss
    
params = (0.01 * np.random.randn(2, NUM_WIRES, 3))
bestparams=(0.01 * np.random.randn(2, NUM_WIRES, 3))
bestcost=1
opt = AdamOptimizer(0.425)
batch_size = 10
Diag_Cost = []
Diag_Acc = []
for it in range(30):
    
    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    X_train_batch = X_train[batch_index]
    Y_train_batch = y_train[batch_index]
    params = opt.step(lambda v: cost(v, X_train_batch, Y_train_batch), params)

    # Compute predictions on train and validation set
    #predictions_train = prediction(params,X_train)

    cosT = cost(params, X_train,y_train)
    # Compute accuracy on train and validation set
    acc = accuracy(params, X_train,y_train) 
    
    if cosT < bestcost:
        bestcost = cosT
        bestparams = params

    Diag_Cost.append(cosT.numpy())
    Diag_Acc.append(acc)
    print(
        "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.2f}% ".format(
        it + 1, cosT, acc*100
    ))

predictions = prediction(bestparams,X_test)
accResult = accuracy(bestparams,X_test,y_test)
print()
print("FINAL ACCURACY: {:0.2f}%".format(accResult*100))