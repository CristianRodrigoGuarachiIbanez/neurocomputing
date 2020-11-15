"""
Created on Fr Nov 13 2020
@author: Cristian Rodrigo Guarachi Ibanez
Multiple Linear Regression
"""

# Multiple linear regression
import numpy as np
from typing import List, Tuple, Dict, Any, TypeVar
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
t = boston.target
print(X.shape)
print(t.shape)
print(boston.DESCR)

plt.figure(figsize=(20, 20))

for i in range(13):
    plt.subplot(4, 4 , i+1)
    plt.scatter(X[:, i], t)
    plt.title(boston.feature_names[i])
plt.show()

#Q7: Apply MLR on the Boston data.
# Print the MSE and visualize the prediction  y  against the true value  t  for each sample as before.
# Does it work?
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def MLR_sklearn(X: int, t: np.ndarray, graphic: bool = None) -> List:

    reg = LinearRegression()
    reg.fit(X, t) #X = 503, 13 und t= 503
    y = reg.predict(X)
    mse = mean_squared_error(y, t)
    # print the MSE y  against the true value  t  for each sample
    print("MSE:", mse)
    if graphic:
        #visualize the prediction y  against the true value  t  for each sample
        plt.plot(t, y, '.')
        plt.plot([t.min(), t.max()], [t.min(), t.max()])
        plt.xlabel("t")
        plt.ylabel("y")
        plt.show()
    else:
        plt.plot(reg.coef_)
        plt.xlabel("Feature")
        plt.ylabel("Weight")
        plt.show()
    return y

y: List = MLR_sklearn(X,t)
#print(y)

from sklearn.preprocessing import scale
X_scaled: scale = scale(X)

# Q8: Apply MLR again on X_scaled, print the MSE and visualize the weights. What has changed?

y: List = MLR_sklearn(X_scaled, t)
#print(y)

#------------------ investigate regularization: ----------------
from sklearn.linear_model import Ridge, Lasso
#MLR with L2 regularization is called Ridge regression
#reg = Ridge(alpha=0.1)
#MLR with L1 regularization is called Lasso regression
#reg = Lasso(alpha=0.1)

def regularizationMLR(X: int, t: np.ndarray, graphic: bool = None, regu: str = "") -> List:
    if not regu:
        reg = LinearRegression()
    elif regu == "ridge":
        reg = Ridge(alpha=0.1)
    elif regu == "lasso":
        reg = Lasso(alpha=0.1)

    reg.fit(X, t) #X = 503, 13 und t= 503
    y = reg.predict(X)
    mse = mean_squared_error(y, t)
    # print the MSE y  against the true value  t  for each sample
    print("MSE regularizated:", mse)
    print("coef:", reg.coef_)
    if graphic:
        #visualize the prediction y  against the true value  t  for each sample
        plt.plot(t, y, '.')
        plt.plot([t.min(), t.max()], [t.min(), t.max()])
        plt.xlabel("t")
        plt.ylabel("y")
        plt.show()
    else:
        plt.plot(reg.coef_)
        plt.xlabel("Feature")
        plt.ylabel("Weight")
        plt.show()
    return y

regu: List = regularizationMLR(X, t, graphic=True, regu="ridge")
#print(regu)