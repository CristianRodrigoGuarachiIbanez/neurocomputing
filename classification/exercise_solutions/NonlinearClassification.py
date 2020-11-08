import numpy as np
import matplotlib.pyplot as plt
from LinearClassification import *

#1.2 Non-linearly separable data

data: np.ndarray = np.loadtxt('nonlinear.csv', delimiter=";")
print(data)
X,t = separateInputOutput(data) #type: np.ndarray, np.ndarray
N,d = X.shape # type: int, int
losses, errors, w, b = perceptron_algorithmus(X,t,N, nb_epochs= 100, eta= 0.1)
visualize(X, t, w, b, title="Nonlinear")
plt.plot(losses, label="losses")
plt.plot(errors, label= "errors")
plt.title("Nonlinearer Datensatz")
plt.legend()
plt.show()


