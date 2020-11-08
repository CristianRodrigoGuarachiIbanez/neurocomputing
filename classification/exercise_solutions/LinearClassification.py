#1 - Perceptron algorithm for linear classification

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Iterable, TypeVar, Any
path: str = "linear.csv"
try:
    data: np.ndarray = np.loadtxt(path, delimiter=";")
except Exception as e:
    print(e)

print(data)
"""data is now a (100, 3) numpy array. The array is organized as follows:

The two first columns are the 2D coordinates of 100 examples (between 0 and 1). This is our input space.
The third column represents the class of each example: +1 for the positive class, -1 for the negative class.
Q1: Create two numpy arrays:

X should be a (100, 2) array containing the inputs.
t should be a (100,) array containing the outputs.
Define also two variables using X.shape: N should have the total number of examples (100) and d should have the number of dimensions of the input space (2)."""
def separateInputOutput(data) -> Tuple[np.ndarray, np.ndarray]:

    X: List = []
    t: List = []

    for row in data:
        x: List = row[:2]
        ti: List = row[2:]
        #print(ti)
        X.append(x)
        t.append(ti)
    t = [col for row in t for col in row] # redefinir la list of list a un simple list
    return np.array(X), np.array(t)
X, t = separateInputOutput(data)
N, d = X.shape # type:int, int
print("Inputs:", X.shape)
print("Outputs:",t)

def visualize(X: np.ndarray, t: np.ndarray, w: np.ndarray = np.array([0., 0.]), b: float = 0.) -> None:
    # Plot the positive and negative examples
    plt.plot(X[t==1., 0], X[t==1., 1], '.')
    plt.plot(X[t==-1., 0], X[t==-1., 1], '.')
    # Plot the hyperplane
    if w[1] != 0.0:
        x = [0., 1.]
        y = [-b/w[1], -(w[0] + b)/w[1]]
        plt.plot(x, y)
    plt.show()


#Q2: Visualize the training set by passing X and t to the visualize() function. Leave the arguments w and b to their default value for now.
plot: Any = visualize(X, t)
"""
The online version of the Perceptron is given by the following algorithm:

Initialize the weight vector w and the bias b.Initialize the weight vector w and the bias b. 
for Mepochs:for Mepochs: 
forall examples (xi,ti):forall examples (xi,ti): 
yi=sign(⟨w.xi⟩+b)yi=sign(⟨w.xi⟩+b) 
w←w+η(ti−yi)xiw←w+η(ti−yi)xi 
b←b+η(ti−yi)b←b+η(ti−yi)"""

#Q3: Let's start by applying the delta learning rule to a single example (for example the first one, with an index i = 0).
## 1.- You will first need to define:
eta: float = 0.1
weight: np.ndarray = np.zeros(2)
bias: int = 0
i: int = 0 # nur die erste Zeile X[i,:]
#2.-die Vorhersage berechnen
#𝑦𝑖=sign(⟨𝐰.𝐱𝑖⟩+𝑏)
y_i: np.ndarray = np.sign(np.dot(weight, X[i,:]) + bias)

#3.- Apply the delta learning to w and b by changing their value.
# Update der Gewichte
#𝐰 ← 𝐰 + 𝜂 * (𝑡𝑖−𝑦𝑖) * 𝐱𝑖
weight += eta * (t[i] - y_i) * X[i, :] #[Zeile:Ziele, Spalte: Spalte] erste Zeile aber alle Spalte

# Update vom Bias
#𝑏 ← 𝑏 + 𝜂*(𝑡𝑖−𝑦𝑖)
bias += eta * (t[i] - y_i)


print("Beispiel", i, ";eta",eta,"; X_i =", X[i, :], "; t_i =", t[i], ";y_i:", y_i)
print("Neue hyperplane: weights =", weight, "; bias =", bias)

#Q4: Modify the preceding code to iterate over all examples of the training set
# (for loop with index i from 0 to N-1).
# The weight vector and the bias should be updated for each example. Visualize the hyperplane at the end of the epoch.

def online_perceptron_algorithm(X,t) -> Tuple:
    """ toma los valores de X:input (100,2) y t: output (100) y calcula este algoritmoperceptron
      para retornar un tuple de los hiperplanos, bias, ademas de la ultima actualización de
      weights y bias"""
    eta: float = 0.1
    weight: np.ndarray = np.zeros(2)
    bias: int = 0
    i: int = 0  # nur die erste Zeile X[i,:]
    tWeights: List = []
    tBias: List = []
    # 2.-die Vorhersage berechnen
    # 𝑦𝑖=sign(⟨𝐰.𝐱𝑖⟩+𝑏)
    for x in X: # oder N, d = X.shape
        y_i: np.ndarray = np.sign(np.dot(weight, x) + bias)

        # 3.- Apply the delta learning to w and b by changing their value.
        # Update der Gewichte
        # 𝐰 ← 𝐰 + 𝜂 * (𝑡𝑖−𝑦𝑖) * 𝐱𝑖
        weight += eta * (t[i] - y_i) * x  # [Zeile:Ziele, Spalte: Spalte] erste Zeile aber alle Spalte
        tWeights.append(weight)
        # Update vom Bias
        # 𝑏 ← 𝑏 + 𝜂*(𝑡𝑖−𝑦𝑖)
        bias += eta * (t[i] - y_i)
        tBias.append(bias)

    return tWeights, tBias, weight, bias

tW, tB, w, b = online_perceptron_algorithm(X, t)

print("Gewicht:", w)
print("Bias:", b)

perceptronView: Any = visualize(X,t,w, b)

#Q5: We now will do nb_epochs = 100 iterations over the training set.
# Edit your code with an additional for loop. What is the hyperplane in the end? Is learning successful?

def online_perceptron_algorithm_epoch(X: np.ndarray,t: np.ndarray, N: int, Epoche: int = 100) -> Tuple:
    """ toma los valores de X:input (100,2) y t: output (100) y calcula este algoritmoperceptron
      para retornar un tuple de los hiperplanos, bias, ademas de la ultima actualización de
      weights y bias"""
    eta: float = 0.1

    weight: np.ndarray = np.zeros(2) # d = 2
    bias: float = 0.

    tWeights: List = []
    tBias: List = []
    # 2.-die Vorhersage berechnen
    # 𝑦𝑖=sign(⟨𝐰.𝐱𝑖⟩+𝑏)
    for _ in range(Epoche):
        #über N iterieren
        for i in range(N): # oder N, d = X.shape
            y_i: np.ndarray = np.sign(np.dot(weight, X[i,:]) + bias)
            #3.- Apply the delta learning to w and b by changing their value.
            # Update der Gewichte
            # 𝐰 ← 𝐰 + 𝜂 * (𝑡𝑖−𝑦𝑖) * 𝐱𝑖
            weight += eta * (t[i] - y_i) * X[i,:]  # [Zeile:Ziele, Spalte: Spalte] erste Zeile aber alle Spalte
            tWeights.append(weight)
            #Update vom Bias
            # 𝑏 ← 𝑏 + 𝜂*(𝑡𝑖−𝑦𝑖)
            bias += eta * (t[i] - y_i)
            tBias.append(bias)
    return tWeights, tBias, weight, bias

W, B, w, b = online_perceptron_algorithm_epoch(X, t, N)
print("Gewicht nach 100 Epoche:", w)
print("Bias nach 100 Epoche:", b)
epoch100: Any = visualize(X,t, w, b)

#Q6: Modify your algorithm to compute the training error and the loss for each epoch.
def perceptron_algorhytmus(X, t, N, d) -> None:

    # Parameters
    eta: int = 0.1
    nb_epochs: int = 100

    # Initialize the weight vector and bias
    w: np.ndarray = np.zeros(d)
    b: float = 0.

    # Perceptron algorithm
    errors: List = []
    losses: List = []
    for epoch in range(nb_epochs):
        misclassification: int = 0
        loss: int = 0

        # Iterate over all training examples
        for i in range(N):
            # Prediction of the hypothesis
            y_i: np.ndarray = np.sign(np.dot(w, X[i, :]) + b)
            # Update the weight
            w += eta * (t[i] - y_i) * X[i, :]
            # Update the bias
            b += eta * (t[i] - y_i)
            # Count misclassifications
            if t[i] != y_i:
                misclassification += 1
            # Loss
            loss += (t[i] - y_i) ** 2 #L(w,b)=  1/N * ∑i=1N(ti−yi)**2

        # Append
        errors.append(misclassification / N)
        losses.append(loss / N)

    visualize(X, t, w, b)
    plt.plot(errors, label="error")
    plt.plot(losses, label="loss")
    plt.legend()
    plt.show()

perceptron_algorhytmus(X, t, N, d)