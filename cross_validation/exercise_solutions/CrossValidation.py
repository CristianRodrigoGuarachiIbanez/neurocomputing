"""
Created on Fr Nov 13 2020
@author: Cristian Rodrigo Guarachi Ibanez
CrossValidation
"""

# Polynomial regression

"""Polynomial regression consists of fitting some data (x,y)(x,y) to a nn-order polynomial of the form:
y=f(x)=w0+w1⋅x+w2⋅x2+...+wn⋅xn
By rewriting the unidimensional input  xx  into the following vector:

x=[1xx2...xn]
x=[1xx2...xn]

and the weight vector as:

w=[w0w1w2...wn]
w=[w0w1w2...wn]

the problem can be reduced to linear regression:

y=⟨w⋅x⟩
y=⟨w⋅x⟩

and we can apply the delta learning rule to find  ww :

Δw=η⋅(ti−yi)⋅xi
Δw=η⋅(ti−yi)⋅xi"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any, TypeVar
#from LinearClassification import separateInputOutput

data: np.ndarray = np.loadtxt("polynome.data", delimiter=";")
X = data[:, 0]
t = data[:, 1]
N = len(X)
print("Cross_Validation")
print(data)

def visualize(X, t, w):
    # Plot the data
    plt.plot(X, t, 'r.')
    # Plot the fitted curve
    x = np.linspace(0., 1., 100)
    y = np.polyval(w, x)
    plt.plot(x, y, 'g-')
    plt.title('Polynomial regression with order ' + str(len(w)-1))
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.show()

#Q1: Apply the np.polyfit() function on the data and visualize the result for different degrees of the polynomial (from 1 to 10 or even more).
# What do you observe? Find a polynomial degree which clearly overfits.

def polyfit(X: np.ndarray, t: np.ndarray, deg: int = 100) -> Any:
    return np.polyfit(X, t, deg)

w: Any = polyfit(X,t)
visualize(X,t,w)

#Q2: Plot the mean square error on the training set for all polynomial regressions from 1 to 10.
# How does the training error evolve when the degree of the polynomial is increased?
# What is the risk by taking the hypothesis with the smallest training error?

def mse(X: np.ndarray, t: np.ndarray, deg: int = 11) -> Tuple[int,np.ndarray, List]:
    training_mse: List = []
    mse: Any = []
    for deg in range(1,deg):
        w: np.polynomial = np.polyfit(X,t,deg)
        y: np.polynomial = np.polyval(w,X)
        mse = np.mean((t-y)**2)
        training_mse.append(mse)
    return deg, mse, training_mse
print("X :", X)
deg,mse_total,training_mse = mse(X,t) # type: int, np.ndarray, List
print("msw:",training_mse)
x_axis: range = range(1, 11)
plt.plot(x_axis, training_mse)
plt.xlabel("Order of the polynomial")
plt.ylabel("Training mse")
plt.show()


#2 - Simple hold-out cross-validation
"""You will now apply simple hold-out cross-validation to find the optimal degree for the polynomial regression. 
You will need to separate the data set into a training set  StrainStrain  (70% of the data) and a test set  StestStest  (the remaining 30%)."""
from sklearn.model_selection import train_test_split
#Q3: Use scikit-learn to split the data into the corresponding training and test sets.

X_train, X_test, t_train, t_test = train_test_split(X,t,test_size = 0.3) # type: np.ndarray, np.ndarray, np.ndarray, np.ndarray

print("X train:", X_train)
print("t train:", t_train)
#Q4: Train each polynomial from degree 1 to 10 on  Strain  and plot the generalization error on  Stest.
# Which degree of the polynomial gives the minimal empirical error? Why?
#Q5: Run the cross-validation split multiple times. Do you always obtain the same optimal degree?
degr, mse_train, train_mse = mse(X_train, t_train) # type: int, np.ndarray, List
x_axis: range = range(1, 11)
plt.plot(x_axis, train_mse)
plt.title("Training set")
plt.xlabel("Order of the polynomial")
plt.ylabel("Training mse")
plt.show()
#3 - k-fold cross-validation

"""As we only have 16 samples, it is quite annoying to "lose" 5 of them for the test set. 
Here we can afford to use k-fold cross-validation, where the cross-validation split is performed  kk  times:"""

from sklearn.model_selection import KFold

k: int = 4
kf: KFold = KFold(n_splits=k, shuffle=True)
print("kf:", kf)

#Q6: Check the doc of KFold (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html).
# Print the indices of the examples of the training and test sets for each iteration of the algorithm. Change the value of  kk  to understand how it works.

for train_index, test_index in kf.split(X, t):
    print("Train_index",train_index)
    print("Train Data:", X[train_index])
    print("Train Ouputs", t[train_index])
    print("Test:",test_index)


#Q7: Apply k-fold cross-validation on the polynomial regression problem.
# Which polynomial degree is the best? Run the split multiple times: does the best polynomial degree change?
k: int = 16
kf: KFold = KFold(n_splits=k, shuffle=True)
test_mse: List = []
for train_index, test_index in kf.split(X,t):
    k_fold: List = []
    degrees, mse_i, trainingSample = mse(X[train_index],t[train_index])
    print(trainingSample) # das ist bereits eine Liste mit dem MSEs
    test_mse.append(trainingSample)

test_mse: np.ndarray = np.mean(test_mse, axis=0)

print(test_mse)

plt.plot(range(1, 11), test_mse)
plt.title("K_fold Cross Validation for Polynomial Regression")
plt.xlabel("Degree of the polynome")
plt.ylabel("k-fold cross-validated mse")
plt.show()


