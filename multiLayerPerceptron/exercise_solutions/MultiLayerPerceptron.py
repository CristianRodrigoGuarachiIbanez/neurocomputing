"""
Created on Fr Nov 13 2020
@author: Cristian Rodrigo Guarachi Ibanez
Multi-layer perception
"""

#Exercise 5 : Multi-layer Perceptron
"""
1 - Structure of the MLP
A.- The output neuron sums its inputs with  KK  weights  W2W2  and a bias  b2b2 . It uses a logistic transfer function:
a) y=σ(∑j=1KW2j⋅hj+b2)

B .- with  σ(x)=1/1+exp−x. As in logistic regression for linear classification, 
we will interpret y as the probability that the input x belongs to the positive class.

C.- Each of the K hidden neurons receives 2 weights from the input layer, what gives a  2×K weight matrix  W1, and K biases  b1. They also use the logistic activation function at first:
c) hj=σ(∑i=12W1i,j * xi+b1j)

D.- The goal is to implement the backpropagation algorithm by comparing the desired output t  with the prediction  y :
Backpropagation of the output error:
d) δ = (t−y) # fordward-pass
d) δj = δ⋅Wj2⋅f′(netj) backward-pass

E.- Parameter updates
e) ΔW1i,j=η⋅δj⋅xi
e) Δb1j=η⋅δj
e) ΔW2j=η⋅δ⋅hj
e) Δb2=η⋅δ
 
F.- You will remember that the derivative of the logistic function is given by:
d) σ′(x)=σ(x)⋅(1−σ(x))"""

#Q1: Why do not we use the derivative of the transfer function of the output neuron when computing the output error  δδ ?
#Answer: As in logistic regression, we will interpret the output of the network as the probability of belonging to the positive class.
# We therefore use (implicitly) the cross-entropy / negative log-likelihood loss function, whose gradient does not include the derivative of the logistic function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Tuple, List, TypeVar, Callable

data: np.ndarray = np.loadtxt("nonlinear.csv")
print(data)


X: np.ndarray = data[:,:2]
t: np.ndarray = data[:,2]

N,d = X.shape # type: int, int

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=.2, random_state=42) #

#Q2: Print the shapes of the different arrays. Observe the values taken by the outputs  t .

eta: float = 0.05 # Learning rate
K: int = 15 # Number of hidden neurons
nb_epochs: int = 2000 # Maximal number of epochs


def uniform_initialization(d: int, K: int, min_val: float, max_val: float) -> Tuple:
    W1: np.random = np.random.uniform(-min_val, max_val, (d, K)) # d is the number of cols
    b1: np.random = np.random.uniform(-min_val, max_val, K)
    W2: np.random = np.random.uniform(-min_val, max_val, K)
    b2: np.random = np.random.uniform(-min_val, max_val, 1)

    return W1, b1, W2, b2


W1, b1, W2, b2 = uniform_initialization(d, K, min_val=0.1, max_val=1.0)

"""The next step is to define the activation/transfer functions. Below are defined the ones we are going to use:

The logistic function  f(x)=1/(1+exp(−x)) .
The tanh function  f(x)=tanh(x) .
The ReLU function  f(x)=max(0,x) ."""

X = TypeVar('X', float, List[float])

def linearFunction(x: X) -> X:
    return x

def logisticFunction(x: X) -> X:
    return 1.0/(1.0 + np.exp(-x))
def tahnFunction(x:X) -> X:
    return np.tanh(x)

def ReLuFunction(x: List[float]) -> List[float]:
    data: List[float] = x.copy()
    data[data < 0.] = 0
    return data

def feedForward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                W2: np.ndarray, b2: np.ndarray, activFunc: Callable) -> Tuple:
    # Hidden Layer
    h: X = activFunc(np.dot(x, W1) + b1)
    # Output Layer
    y: X = logisticFunction(np.dot(h, W2) + b2)
    return h,y

# Q3. : Using the randomly initialized weights, apply the feedforward() method to an input vector (for example  [0.5,0.5][0.5,0.5] ) and print h and y.
# What is the predicted class of the example?

input: List = [[0.5,0.5],[0.5,0.5]]
input: np.ndarray = np.array(input)

hidden, output = feedForward(input[0], W1, b1, W2, b2, logisticFunction) #type: X, X

print(hidden,output)
if output > 0.5:
    print('Positive Class')
else:
    print('Negative Class')

#Q4: Plot the initial classification for both datasets (you will need to re-run the relevant cells).
# Is there a need for learning?
# Reinitialize the weights and biases multiple times. What do you observe?

def plot_classification(X_train: np.ndarray, t_train: np.ndarray, X_test: np.ndarray,
                        t_test: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
                        activation_function: Callable):
    #print("X train", type(X_train))
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max() # type: float, float
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max() # type: float, float
    #print("x min", x_min)
    # put the the data in mesh grid after aranging it from ... to ... by taking steps ...
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02)) # type: np.ndarray, np.ndarray
    #print("xx", xx)
    Z: X = feedForward(np.c_[xx.ravel(), yy.ravel()], W1, b1, W2, b2, activation_function)[1]
    Z[Z > 0.5] = 1
    Z[Z <= 0.5] = 0
    #print("Z", Z)
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    #to plot contours. But contourf draw filled contours, while contourf draws contour lines.
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm_bright, alpha=.4) # put the data in the plot

    plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train, cmap=cm_bright, edgecolors='k')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=t_test, cmap=cm_bright, alpha=0.4, edgecolors='k')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


plot_classification(X_train, t_train, X_test, t_test, W1, b1, W2, b2, logisticFunction)


#Q5: Implement the online backpropagation algorithm to classify nonlinear.csv.

