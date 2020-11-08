import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from typing import List, Tuple, Dict, Any, TypeVar
#1 - Least mean squares: pip install scikit-learn

N:int = 100
X, t = make_regression(n_samples=N, n_features=1, noise=15.0)
print(X.shape)
print(t.shape)
plt.scatter(X, t)
plt.title("Data for Regression")
plt.xlabel("x")
plt.ylabel("t")
plt.show()
#Q2: Implement the LMS algorithm and apply it to the generated data.
# The Python code that you will write is almost a line-by-line translation of the pseudo-code above.
# You will use a learning rate eta = 0.1 at first, but you can choose another value later.
# Start by running a single epoch, as it will be easier to debug it, and then increase the number of epochs to 100 or so. Print the value of the weight and bias at the end.

"""Remember the LMS algorithm from the course:"""

def lms_algorythmus(w):
  i: int = 0
  alpha: list = [0.1, 0.01, 0.001] #lernrate
  Emax:int = 10000000000000000000000000
  maxIter:int = 500
  E: int  = 0
  while ((i < maxIter) and (E < Emax)):
    E = 0
    for pair in training:
      y = np.dot(w.transpose(), pair[0])
      #print('y: ')
      #print(y)
      #print('==================')
      w = w + np.dot((alpha[2]*(pair[1] - y)), pair[0])
      #print('weight: ' + str(w))
      E = E + np.power((pair[1] - y), 2)
      #print('Error: ' + str(E))
    #Put the error in the array after going thru the whole training pattern
    error_grp.append(E)
    iter_grp.append(i)
    i = i + 1
  #print('i: ' + str(i))

  print('Final error: ' + str(E))
  print('final weight: ' + str(w))
def LMS_algorythmus(E: int, N: int, X: np.ndarray, t: np.ndarray) -> Tuple[int, int]:
    '''nimmt eine Anzahl an Wiederholungen bzw. Epochen "E" und die Stichprobegröße "N" und gibt
      die angepassten Gewichte "weight" und Verzerrung "bias" '''
    weight: int = 0 #w=0;b=0w
    bias: int = 0

    eta: float = 0.1

    for epoch in range(E): #for E epochs:
        dweight: int = 0  # dw=0; db=0
        dbias: float = 0.0
        for i in range(N):  #for each sample(xi,ti) :
            y: np.ndarray = weight * X[i] + bias  #yi = w*xi+b
            dweight += (t[i] - y) * X[i] # dw = dw+(ti−yi)*xi
            dbias += (t[i] - y)          # db = db+(ti−yi)

        weight += eta * dweight / N # Δw = η * 1/N * dw
        bias += eta * dbias / N  # Δb=η * 1/N * db

    return weight, bias
#Q3: Visualize the quality of the fit by superposing the learned model to the data with matplotlib.
W, b = LMS_algorythmus(100,N,X,t)
plt.scatter(X,t) # learned model
x_axis: list = [X.min(),X.max()]
plt.plot(x_axis, W*x_axis + b )
plt.title("used LMS Algorythm withount MSE")
plt.xlabel("X_Axis")
plt.ylabel("t_axis")
plt.show()

#Q4: Make a scatter plot where  t  is the x-axis and  y=w∗X+b is the y-axis.
# How should the points be arranged in the ideal case? Also plot what this ideal relationship should be.
y: np.ndarray = W*X+b

plt.scatter(t, y) # im Scatter 1: Zeile/x_axis , 2: Spalter/y_axis
x_axis = [t.min(), t.max()]
plt.plot(x_axis, x_axis)
plt.title("used LSM Algorythm")
plt.xlabel("t")
plt.ylabel("y")
plt.show()

#Q5: Modify your LMS algorithm (either directly or copy it in the next cell) to track the MSE after each epoch.
# After each epoch, append the MSE on the training set to a list and plot it at the end.
# How does the MSE evolve? Which value does it get in the end? Why? How many epochs do you actually need?
###MSEx = 1/N sum(t_i - y_i)^2 -> sum über N i=1

def LMS_MSE_algorythmus(E, N, graphic:bool = False) -> Tuple[int, int, List]:
    weight:int = 0
    bias: int = 0

    eta: float = 0.1
    #sammelt jedes MSE Ergebnis
    losses: list = []
    for epoch in range(E):
        dweight: int = 0
        dbias: int = 0.0
        mse: float = 0.0
        for i in range(N):
            y: float = weight * X[i] + bias
            dweight += (t[i] - y) * X[i]
            dbias += (t[i] - y)
            mse += (t[i] - y) ** 2
        weight += eta * dweight / N
        bias += eta * dbias / N
        losses.append(mse / N)
    if graphic:

        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.show()
    return weight, bias, losses

w,b,mse = LMS_MSE_algorythmus(100, N, True)
print(mse)
# Q6: Apply linear regression on the data using scikit-learn.
# Check the model parameters after learning and compare them to what you obtained previously.
# Make a plot comparing the predictions with the data.
def linearRegression_scikit_learn(X: int, t: int):
    from sklearn.linear_model import LinearRegression
    regression: LinearRegression = LinearRegression()
    regression.fit(X,t)

    return regression.coef_, regression.intercept_, regression.predict(X)

coef, inter, y = linearRegression_scikit_learn(X, t)
plt.scatter(t, y)
x_axis = [t.min(), t.max()]
plt.plot(x_axis, x_axis)
plt.title("Plot Data Sklearn")
plt.xlabel("t")
plt.ylabel("y")
plt.show()