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
from typing import Tuple, List, TypeVar, Callable, Any
from tabulate import tabulate

# data: np.ndarray = np.loadtxt("nonlinear.csv")
# print(data)
#
# X: np.ndarray = data[:,:2]
# t: np.ndarray = data[:,2]
#
# N,d = X.shape # type: int, int

def prepareData(filename: str, meanRemoval: bool = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """ Prepare the data """
    data: np.ndarray = np.loadtxt(filename)
    X: np.ndarray = data[:, :2]
    t: np.ndarray = data[:, 2]
    N, d = X.shape  # type: int, int

    if(meanRemoval):
        # Mean removal
        X -= X.mean()

    return X, t, N, d
#X, t, N, d = prepareData("nonlinear.csv")

#X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=.2, random_state=42) #

#Q2: Print the shapes of the different arrays. Observe the values taken by the outputs  t .

eta: float = 0.05 # Learning rate
K: int = 15 # Number of hidden neurons
nb_epochs: int = 2000 # Maximal number of epochs


def uniform_initialization(d: int, K:int, min_val: float, max_val: float) -> Tuple:
    W1: np.random = np.random.uniform(-min_val, max_val, (d, K)) # d ist num. of rows and k is the number of cols
    b1: np.random = np.random.uniform(-min_val, max_val, K)
    W2: np.random = np.random.uniform(-min_val, max_val, K)
    b2: np.random = np.random.uniform(-min_val, max_val, 1)

    return W1, b1, W2, b2


#W1, b1, W2, b2 = uniform_initialization(d, K, min_val=1.0, max_val=1.0)

"""The next step is to define the activation/transfer functions. Below are defined the ones we are going to use:

The logistic function  f(x)=1/(1+exp(−x)) .
The tanh function  f(x)=tanh(x) .
The ReLU function  f(x)=max(0,x) ."""

X = TypeVar('X', float, List[float], Tuple[np.ndarray])

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
                W2: np.ndarray, b2: np.ndarray, activFunc: Callable) -> Tuple[X,X]:
    # Hidden Layer
    h: X = activFunc(np.dot(x, W1) + b1)
    # Output Layer
    y: X = logisticFunction(np.dot(h, W2) + b2)
    return h,y

# Q3. : Using the randomly initialized weights, apply the feedforward() method to an input vector (for example  [0.5,0.5][0.5,0.5] ) and print h and y.
# What is the predicted class of the example?

#input: List = [[0.5,0.5],[0.5,0.5]]
#input: np.ndarray = np.array(input)

#hidden, output = feedForward(input[0], W1, b1, W2, b2, logisticFunction) #type: X, X

#print(hidden,output)
#if output > 0.5:
#    print('Positive Class')
#else:
#    print('Negative Class')

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


#plot_classification(X_train, t_train, X_test, t_test, W1, b1, W2, b2, logisticFunction)


#Q5: Implement the online backpropagation algorithm to classify nonlinear.csv.
"""All you have to do is to implement the backpropagation algorithm and adapt the parameters after each example:

compute the output error delta
compute the backpropagated error delta_hidden
increment the parameters W1, b1, W2, b2 accordingly"""
def backpropagation(X_train: np.ndarray, t_train: np.ndarray, activation_function: Callable, d: int, K: int, eta: float,
                    nb_epochs: int, weight_init: float, plotting: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:

    W1, b1, W2, b2 = uniform_initialization(d, K, min_val=weight_init, max_val=weight_init)  # type: X, X,X,X

    errors = []
    for epoch in range(nb_epochs):
        nb_errors = 0
        for i in range(X_train.shape[0]):

            # Feedforward pass
            h, y = feedForward(X_train[i, :], W1, b1, W2, b2, activation_function) #type: X, X #wert zw. 0 und 1

            # Predict the class:
            if y > 0.5:
                c = 1
            else:
                c = 0

            # Count the number of misclassifications
            if t_train[i] != c: # if das gewünschte Output anders als das vorhergesagte Output
                nb_errors += 1  # dann füght eine Missclassification an
                print("Missclassification value:", nb_errors)

            # TODO: backpropagation and parameter updates
            # Output Error
            delta: X = (t_train[i] - y)

            # Hidden error w2 * delta * 'f(h) wobei f(h) = f(h)*(1. - h)
            delta_hidden: float = W2 * delta * h * (1. - h) # logistische function aber as backpropagated transfer function


            # Learn the output weights
            W2 += eta * delta * h

            # Learn the output bias
            b2 += eta * delta

            # Learn the hidden weights
            W1 += eta * np.outer(X_train[i, :], delta_hidden) # eta * delta_hidden * xT
            print("was ist x hier:", X_train[i, :])

            # Learn the hidden biases
            b1 += eta * delta_hidden


        # Compute the error rate
        errors.append(nb_errors / X_train.shape[0]) # dar letzte Missklassifisierungswert / N

        # Stop when there are not errors
        if nb_errors == 0:
            break

    print('Number of epochs needed:', epoch + 1)
    print('Training accuracy:', 1 - errors[-1])
    print('Missclassification rate:', errors)
    if(plotting):
        plot_classification(X_train, t_train, X_test, t_test, W1, b1, W2, b2, logisticFunction)
        plt.plot(errors)
        plt.xlabel("Number of epochs")
        plt.ylabel("Training error")
        plt.show()
    return W1, b1, W2, b2, errors


#Q6: Create a method test_accuracy(X_test, t_test, W1, b1, W2, b2, func)
# that returns the accuracy of the model on the test set.
# What is the test accuracy of your network after training? Compare it to the training accuracy.

def test_accuracy(X_test: np.ndarray, t_test: np.ndarray, W1: np.ndarray,
                  b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, activation_function: Callable) -> float:
    nb_errors: int = 0
    for i in range(X_test.shape[0]):

        # Feedforward pass
        h, y = feedForward(X_test[i, :], W1, b1, W2, b2, activation_function) # type: np.ndarray, np.ndarray

        # Predict the class:
        if y > 0.5:
            c = +1
        else:
            c = 0

        # Count the number of misclassifications
        if t_test[i] != c:
            nb_errors += 1

    return 1 - nb_errors / X_test.shape[0]

#vorher uniform_initialization definieren
#print("Test accuracy:", test_accuracy(X_test, t_test, W1, b1, W2, b2, logisticFunction))

#Q7: To avoid copy and pasting too much, create a function train_MLP taking the training set as input,
# some hyperparameters (K, eta...) and return the weights and returning the final parameters as well as the history of errors:


#X, t, N, d = prepareData('nonlinear.csv')

# Create training and test sets
#X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=.2, random_state=42)

#hyperparameter
#eta_7: float = 0.05
#K_7: int = 15
# nb_epochs_7: int = 10
# activation_function_7: Any = logisticFunction
# weight_init_7: float = 1.0

# Train the network
#W1, b1, W2, b2, errors = backpropagation(X_train, t_train, activation_function, d, K, eta, nb_epochs, weight_init) #type: np.array, np.ndarray, np.ndarray, List

# print('Number of epochs needed:', len(errors))
# print('Training accuracy:', 1-errors[-1])
# print("Test accuracy:", test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function))
# plot_classification(X_train, t_train, X_test, t_test, W1, b1, W2, b2, activation_function)
# plt.plot(errors)
# plt.title("Nonlinear training data")
# plt.xlabel("Number of epochs")
# plt.ylabel("Training error")
# plt.show()

#Q8: Apply your algorithm on circular.csv. What changes?

# Prepare the data
def PrepareTrainTestData(filename: str, meanRemoval: bool = None) -> Tuple:
    X, t, N, d = prepareData(filename, meanRemoval)
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=.2, random_state=42)
    return X_train, X_test, t_train, t_test, d
#X_train, X_test, t_train, t_test = PrepareTrainTestData("circular.csv")
# Train the network
#W1, b1, W2, b2, errors = backpropagation(X_train, t_train, activation_function, d, K, eta, nb_epochs, weight_init) #type: np.array, np.ndarray, np.ndarray, List

# print('Number of epochs needed:', len(errors))
# print('Training accuracy:', 1-errors[-1])
# print("Test accuracy:", test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function))
# plot_classification(X_train, t_train, X_test, t_test, W1, b1, W2, b2, activation_function)
# plt.plot(errors)
# plt.title("Circular data")
# plt.xlabel("Number of epochs")
# plt.ylabel("Training error")
# plt.show()


#Q9: Try different values for the number of hidden neurons  KK  (e.g. 2, 5, 10, 15, 20, 25, 50...)
# and observe how the accuracy and speed of convergence evolve for both datasets.

#K_9: Tuple[int,...] = (2, 5, 10, 15, 20, 25, 50)

#acurracy_results_9: List = [["index", "hidden neurons", "result"]]
#i: int = 1
# for k in K:
#     W1, b1, W2, b2, errors = backpropagation(X_train, t_train, activation_function, d, k, eta, nb_epochs, weight_init) #type: np.array, np.ndarray, np.ndarray, List
#     print(f"acurracy with {k} layers")
#     acurracy_results.append([i, k, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function = activation_function)])
#     i += 1
# print(acurracy_results)
"""Answer: Surprisingly, 2 hidden neurons are enough for the non-linear dataset (with some remaining errors) 
and 3 for the circular dataset. These problems are really easy.
The more hidden neurons, the faster it converges (in terms of epochs, not computation time...)"""

#Q10: The weights are initialized randomly between -1 and 1. Try to initialize them to 0. Does it work? Why?
#nb_epochs = 100

# X_train, X_test, t_train, t_test, d = PrepareTrainTestData("circular.csv")
#acurracy_results_10: List = [["index", "hidden neurons", "result"]]
#
# for k in K:
#     W1, b1, W2, b2, errors_10 = backpropagation(X_train, t_train, activation_function, d, k, eta, nb_epochs, weight_init = 0.) #type: np.array, np.ndarray, np.ndarray, List
#     print(f"acurracy with {k_10} layers")
#     acurracy_results_10.append([i, k_10, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function = activation_function)])
#     i += 1
# print(acurracy_results_10)

"""Answer: this is a simple example of vanishing gradient. The backpropagated error gets multiplied by W2, which is initially zero. 
There is therefore no backpropagated error, the first layer cannot learn anything for a quite long time. 
The global function stays linear."""

#Q11: Vary the learning rate between extreme values (with weights again randomly initialized).
# How does the performance evolve?
# X_train, X_test, t_train, t_test, d = PrepareTrainTestData("circular.csv")
#
# #hyperparameter
# k: int = 2
# eta: float = 0.2
# nb_epochs: int = 10
# activation_function: Any = logisticFunction
# weight_init_11: float = 1.0
#
# W1, b1, W2, b2, errors = backpropagation(X_train, t_train, activation_function, d, k, eta, nb_epochs, weight_init=weight_init_11, plotting= False)  # type: np.array, np.ndarray, np.ndarray, List
# print(f"acurracy with {k} layers and eta {eta} for circular")
# print(tabulate([["Num. Hidden Layers", "Acurracy Rate"],[k, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function=activation_function)]]
#                , headers="firstrow", showindex="always", tablefmt="github"))
#
# X_train, X_test, t_train, t_test, d = PrepareTrainTestData("nonlinear.csv")
# W1, b1, W2, b2, errors = backpropagation(X_train, t_train, activation_function, d, k, eta, nb_epochs, weight_init=weight_init_11, plotting=False)  # type: np.array, np.ndarray, np.ndarray, List
# print(f"acurracy with {k} layers and eta {eta} for nonlinear")
# print(tabulate([["Num. Hidden Layers", "Acurracy Rate"],[k, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function=activation_function)]]
#                , headers="firstrow", showindex="always", tablefmt="github"))

#Q12: For a fixed number of hidden neurons (e.g.  K=15 ) and a correct value of eta, run 10 times the same network.
# Does the performance change? What is the mean number of epochs needed to get a training error of 0?

#E = TypeVar("E", List, int)

# #hyperparameter
# k: int = 15
# eta: float = 0.2
# nb_epochs: int = 200
# activation_function: Any = logisticFunction
# weight_init_12: float = 1.0

#prepare data and create train test data
#X_train, X_test, t_train, t_test, d = PrepareTrainTestData("nonlinear.csv")

#X_train, X_test, t_train, t_test, d = PrepareTrainTestData("circular.csv")

# # train the network for 10 times
# performance: List = []
# acurracy_results_12: List = [["index","Num. Hidden Layers", "Acurracy Rate"]]
# for i in range(10):
#     W1, b1, W2, b2, errors = backpropagation(X_train, t_train, activation_function, d, k, eta, nb_epochs, weight_init=weight_init_11, plotting=False)  # type: np.array, np.ndarray, np.ndarray, List
#     print(f"acurracy with {k} layers and eta {eta} for nonlinear")
#     acurracy_results_12.append([i,k, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function=activation_function)])
#     nb_epochs = len(errors) # length des Training Errors
#     #print(len(errors))
#     performance.append(nb_epochs)
#
# print(tabulate(acurracy_results_12, headers="firstrow", showindex="always", tablefmt="github"))
#
# plt.plot(performance)
# plt.xlabel("Trial")
# plt.ylabel("Number of epochs to convergence")
# plt.show()

#Q13: Modify the train_mlp() method so that it applies backpropagation correctly for any of the four activation functions:

#linear:  f′(x) = 1
#logistic:  f′(x) = f(x)(1−f(x))
#tanh:  f′(x) =(1−f2(x))
#relu:  f′(x)={1 if x > 0logisticFunction
#              0 if x ≤ 0

def train_mlp(X_train: np.ndarray, t_train: np.ndarray, activation_function: Callable, d: int, K: int, eta: float,
                    nb_epochs: int, weight_init: float, plotting: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    L = TypeVar("L", List, float)
    W1, b1, W2, b2 = uniform_initialization(d, K, min_val=weight_init, max_val=weight_init)  # type: X, X,X,X

    errors = []
    for epoch in range(nb_epochs):
        nb_errors = 0
        for i in range(X_train.shape[0]):

            # Feedforward pass
            h, y = feedForward(X_train[i, :], W1, b1, W2, b2, activation_function) #type: X, X #wert zw. 0 und 1
            print("Previous Hidden Layer:", h)
            # Predict the class:
            if y > 0.5:
                c = 1
            else:
                c = 0

            # Count the number of misclassifications
            if t_train[i] != c: # if das gewünschte Output anders als das vorhergesagte Output
                nb_errors += 1  # dann füght eine Missclassification an
                #print("Missclassification value:", nb_errors)

            # TODO: backpropagation and parameter updates
            # Output Error
            delta: X = (t_train[i] - y)

            #transfer function
            transferFunc: Callable = Any
            if( activation_function == linearFunction):
                transferFunc: float = h
            elif(activation_function == logisticFunction):
                transferFunc: float = h * (1. - h)
            elif(activation_function == ReLuFunction):
                #func: List[int] = [1 if x > 0. else 0 for x in h if x <= 0.]
                #func: np.ndarray = np.array(func)
                transferFunc: List[float] = h.copy()
                transferFunc[transferFunc > 0.0] = 1.0
                transferFunc[transferFunc <= 0.0] = 0.0
            elif(activation_function == tahnFunction):
                transferFunc: float = 1.0 - h * h

            print("Hidden Layer:",h, "transfer Func:", transferFunc)

            # Hidden error
            delta_hidden: L = W2 * delta * transferFunc

            # Learn the output weights
            W2 += eta * delta * h

            # Learn the output bias
            b2 += eta * delta

            # Learn the hidden weights
            W1 += eta * np.outer(X_train[i, :], delta_hidden) # eta * delta_hidden * xT
            print("was ist x hier:", X_train[i, :])

            # Learn the hidden biases
            b1 += eta * delta_hidden


        # Compute the error rate
        errors.append(nb_errors / X_train.shape[0]) # der letzte Missklassifisierungswert / N

        # Stop when there are not errors
        if nb_errors == 0:
            break

    print('Number of epochs needed:', epoch + 1)
    print('Training accuracy:', 1 - errors[-1])
    print('Activation function:', activation_function)
    if(plotting):
        plot_classification(X_train, t_train, X_test, t_test, W1, b1, W2, b2, logisticFunction)
        plt.plot(errors)
        plt.xlabel("Number of epochs")
        plt.ylabel("Training error")
        plt.show()
    return W1, b1, W2, b2, errors

#X_train, X_test, t_train, t_test, d = PrepareTrainTestData("nonlinear.csv")

# X_train, X_test, t_train, t_test, d = PrepareTrainTestData("circular.csv")
#
# #hyperparameter
# k_13: int = 15
# eta_13: float = 0.2
# nb_epochs_13: int = 200
# activation_function_13: Any = ReLuFunction
# weight_init_13: float = 1.0
#
# acurracy_results_13: List = [["Num. Hidden Layers", "Acurracy Rate"]]
# W1, b1, W2, b2, errors_13 = train_mlp(X_train, t_train, activation_function_13, d, k_13, eta, nb_epochs_13, weight_init=weight_init_13, plotting=False)  # type: np.array, np.ndarray, np.ndarray, List
# acurracy_results_13.append([k_13, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function=activation_function_13)])
# print(tabulate(acurracy_results_13, headers="firstrow", showindex="always", tablefmt="github"))

#Q14: Use a linear transfer function for the hidden neurons.
# How does performance evolve? Is the non-linearity of the transfer function important for learning?

# # Hyperparameters
# eta_14: float = 0.2
# K_14: int = 15
# nb_epochs_14: int = 200
# #activation_function_14: Callable = linearFunction #tahnFunction
# weight_init_14 = 1.0
#
# X_train, X_test, t_train, t_test, d_14 = PrepareTrainTestData("nonlinear.csv")#circular.csv")

#Q15: Use this time the hyperbolic tangent function as a transfer function for the hidden neurons. 
# Does it improve learning?
#activation_function_14: Callable = tahnFunction

#Q16: Use the Rectified Linear Unit (ReLU) transfer function.
# What does it change? Conclude on the importance of the transfer function for the hidden neurons.
# Select the best one from now on.

#Hyperparameter
#activation_function_14: Callable = ReLuFunction

# acurracy_results_13: List = [["Num. Hidden Layers", "Acurracy Rate"]]
# W1, b1, W2, b2, errors_14 = train_mlp(X_train, t_train, activation_function_14, d_14, K_14, eta, nb_epochs_14, weight_init=weight_init_14, plotting=True)  # type: np.array, np.ndarray, np.ndarray, List
# acurracy_results_13.append([K_14, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function=activation_function_14)])

#Q17: In order to improve the convergence speed, try:

#to remove the mean value from the inputs (X -= np.mean(X) before splitting it into training and test sets).
#to randomize the order in which the examples are presented during one epoch (check the doc for the Numpy method np.random.permutation()) in train_MLP().
#Does it improve convergence?

# def train_MLP(X_train: np.ndarray, t_train: np.ndarray, activation_function: Callable, d: int, K: int, eta: float,
#                     nb_epochs: int, weight_init: float, plotting: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
#     L = TypeVar("L", List, float)
#     W1, b1, W2, b2 = uniform_initialization(d, K, min_val=weight_init, max_val=weight_init)  # type: X, X,X,X
#
#     errors = []
#     for epoch in range(nb_epochs):
#         nb_errors = 0
#         for i in np.random.permutation(X_train.shape[0]): # randomize the order of samples
#
#             # Feedforward pass
#             h, y = feedForward(X_train[i, :], W1, b1, W2, b2, activation_function) #type: X, X #wert zw. 0 und 1
#             print("Previous Hidden Layer:", h)
#             # Predict the class:
#             if y > 0.5:
#                 c = 1
#             else:
#                 c = 0
#
#             # Count the number of misclassifications
#             if t_train[i] != c: # if das gewünschte Output anders als das vorhergesagte Output
#                 nb_errors += 1  # dann füght eine Missclassification an
#                 #print("Missclassification value:", nb_errors)
#
#             # TODO: backpropagation and parameter updates
#             # Output Error
#             delta: X = (t_train[i] - y)
#
#             #transfer function
#             transferFunc: Callable = Any
#             if( activation_function == linearFunction):
#                 transferFunc: float = h
#             elif(activation_function == logisticFunction):
#                 transferFunc: float = h * (1. - h)
#             elif(activation_function == ReLuFunction):
#                 #func: List[int] = [1 if x > 0. else 0 for x in h if x <= 0.]
#                 #func: np.ndarray = np.array(func)
#                 transferFunc: List[float] = h.copy()
#                 transferFunc[transferFunc > 0.0] = 1.0
#                 transferFunc[transferFunc <= 0.0] = 0.0
#             elif(activation_function == tahnFunction):
#                 transferFunc: float = 1.0 - h * h
#
#             print("Hidden Layer:",h, "transfer Func:", transferFunc)
#
#             # Hidden error
#             delta_hidden: L = W2 * delta * transferFunc
#
#             # Learn the output weights
#             W2 += eta * delta * h
#
#             # Learn the output bias
#             b2 += eta * delta
#
#             # Learn the hidden weights
#             W1 += eta * np.outer(X_train[i, :], delta_hidden) # eta * delta_hidden * xT
#             print("was ist x hier:", X_train[i, :])
#
#             # Learn the hidden biases
#             b1 += eta * delta_hidden
#
#
#         # Compute the error rate
#         errors.append(nb_errors / X_train.shape[0]) # der letzte Missklassifisierungswert / N
#
#         # Stop when there are not errors
#         if nb_errors == 0:
#             break
#
#     print('Number of epochs needed:', epoch + 1)
#     print('Training accuracy:', 1 - errors[-1])
#     print('Activation function:', activation_function)
#     if(plotting):
#         plot_classification(X_train, t_train, X_test, t_test, W1, b1, W2, b2, logisticFunction)
#         plt.plot(errors)
#         plt.xlabel("Number of epochs")
#         plt.ylabel("Training error")
#         plt.show()
#     return W1, b1, W2, b2, errors

# #Create training and test sets
# X_train, X_test, t_train, t_test, d_17 = PrepareTrainTestData("nonlinear.csv", meanRemoval = True)#circular.csv")
#
# # Hyperparameters
# eta_17: float = 0.1
# K_17: int = 15
# nb_epochs_17: int = 2000
# activation_function_17: Callable = ReLuFunction
# weight_init_17: float = 1.0
#
# # Train the network
# acurracy_results_17: List = []
# W1, b1, W2, b2, errors = train_MLP(X_train, t_train, activation_function_17, d_17, K_17, eta_17, nb_epochs_17, weight_init_17, plotting=True)
#
# acurracy_results_17.append([K_17, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function=activation_function_17)])
#
# print(errors)
# print(acurracy_results_17)

#Q18: According to the empirical analysis by Glorot and Bengio in “Understanding the difficulty of training deep feedforward neural networks”,
# the optimal initial values for the weights between two layers of a MLP are uniformly taken in the range:
#[−√6/ Nin+Nout ; √ 6/ Nin+Nout]

#where  Nin  is the number of neurons in the first layer and  Nout  the number of neurons in the second layer.

#Create a glorot_initialization() method similar to uniform_initialization() to initialize both hidden and output weights with this new range.
#The biases should be initialized to 0. Use this initialization in train_MLP. Does it help?


def glorot_initialization(d: int, k: int) -> Tuple:

    max_value: float = np.sqrt(6./(d+k))
    W1: np.ndarray = np.random.uniform(-max_value, max_value, (d,k))
    b1: np.ndarray = np.zeros(k)
    max_value = np.sqrt(6. /(k+1))
    W2: np.ndarray = np.random.uniform(-max_value, max_value, k)
    b2: np.ndarray = np.zeros(1)

    return W1, b1, W2, b2

def train_MLP(X_train: np.ndarray, t_train: np.ndarray, activation_function: Callable, d: int, K: int, eta: float,
                    nb_epochs: int, weight_init: float, plotting: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    L = TypeVar("L", List, float)
    W1, b1, W2, b2 = glorot_initialization(d, K)  # type: X, X, X, X

    errors = []
    for epoch in range(nb_epochs):
        nb_errors = 0
        for i in np.random.permutation(X_train.shape[0]): # randomize the order of samples

            # Feedforward pass
            h, y = feedForward(X_train[i, :], W1, b1, W2, b2, activation_function) #type: X, X #wert zw. 0 und 1
            print("Previous Hidden Layer:", h)
            # Predict the class:
            if y > 0.5:
                c = 1
            else:
                c = 0

            # Count the number of misclassifications
            if t_train[i] != c: # if das gewünschte Output anders als das vorhergesagte Output
                nb_errors += 1  # dann füght eine Missclassification an
                #print("Missclassification value:", nb_errors)

            # TODO: backpropagation and parameter updates
            # Output Error
            delta: X = (t_train[i] - y)

            #transfer function
            transferFunc: Callable = Any
            if( activation_function == linearFunction):
                transferFunc: float = h
            elif(activation_function == logisticFunction):
                transferFunc: float = h * (1. - h)
            elif(activation_function == ReLuFunction):
                #func: List[int] = [1 if x > 0. else 0 for x in h if x <= 0.]
                #func: np.ndarray = np.array(func)
                transferFunc: List[float] = h.copy()
                transferFunc[transferFunc > 0.0] = 1.0
                transferFunc[transferFunc <= 0.0] = 0.0
            elif(activation_function == tahnFunction):
                transferFunc: float = 1.0 - h * h

            print("Hidden Layer:",h, "transfer Func:", transferFunc)

            # Hidden error
            delta_hidden: L = W2 * delta * transferFunc

            # Learn the output weights
            W2 += eta * delta * h

            # Learn the output bias
            b2 += eta * delta

            # Learn the hidden weights
            W1 += eta * np.outer(X_train[i, :], delta_hidden) # eta * delta_hidden * xT
            print("was ist x hier:", X_train[i, :])

            # Learn the hidden biases
            b1 += eta * delta_hidden


        # Compute the error rate
        errors.append(nb_errors / X_train.shape[0]) # der letzte Missklassifisierungswert / N

        # Stop when there are not errors
        if nb_errors == 0:
            break

    print('Number of epochs needed:', epoch + 1)
    print('Training accuracy:', 1 - errors[-1])
    print('Activation function:', activation_function)
    if(plotting):
        plot_classification(X_train, t_train, X_test, t_test, W1, b1, W2, b2, logisticFunction)
        plt.plot(errors)
        plt.xlabel("Number of epochs")
        plt.ylabel("Training error")
        plt.show()
    return W1, b1, W2, b2, errors

#Create training and test sets
X_train, X_test, t_train, t_test, d_18 = PrepareTrainTestData("circular.csv", meanRemoval = True)#circular.csv")

# Hyperparameters
eta_18: float = 0.1
K_18: int = 15
nb_epochs_18: int = 20
activation_function_18: Callable = ReLuFunction
weight_init_18: float = 1.0

# Train the network
acurracy_results_18: List = []
W1, b1, W2, b2, errors_18 = train_MLP(X_train, t_train, activation_function_18, d_18, K_18, eta_18, nb_epochs_18, weight_init_18, plotting=True)

acurracy_results_18.append([K_18, test_accuracy(X_test, t_test, W1, b1, W2, b2, activation_function=activation_function_18)])

print("training errors:",errors_18)
print("acurracy values:",acurracy_results_18)


"""quite small changes can drastically change the performance of the network, both in terms of accuracy and training time.
 We went from 2000 epochs to converge to 100 or less. 
Mean removal, Glorot initialzation and the use of ReLU are the most determinant here"""