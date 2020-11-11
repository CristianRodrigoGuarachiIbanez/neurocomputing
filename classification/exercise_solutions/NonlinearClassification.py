import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, TypeVar
from LinearClassification import separateInputOutput, perceptronOnlineVersion, visualize

#1.2 Non-linearly separable data

data: np.ndarray = np.loadtxt('nonlinear.csv', delimiter=";")
print(data)
X,t = separateInputOutput(data) #type: np.ndarray, np.ndarray
N,d = X.shape # type: int, int


losses, errors, w, b = perceptronOnlineVersion(X,t,N, nb_epochs= 100, eta= 0.1)
visualize(X, t, w, b, title="Nonlinear")
plt.plot(losses, label="losses")
plt.plot(errors, label= "errors")
plt.title("Nonlinearer Datensatz")
plt.legend()
plt.show()


#1.3 - Batch version
"""Let's now implement the batch version of the Perceptron algorithm for comparison. The algorithm is theoretically:

Initialize the weight vector w and the bias b.Initialize the weight vector w and the bias b. 
for Mepochs:for Mepochs: 
Initialize weight and bias changes Δw=0,Δb=0Initialize weight and bias changes Δw=0,Δb=0 
forall examples (xi,ti):
 
  yi=sign(⟨w.xi⟩+b)
  Δw←Δw+(ti−yi)xi 
  Δb←Δb+(ti−yi) 
  w←w+ηΔw/N 
  b←b+ηΔb/N"""
#Q10: Reload the linear.csv dataset, initialize the weight vector and bias as usual, and print the shape of the prediction.
#X, und t sind bereits definiert
w: np.ndarray = np.zeros(d)
b: float = 0.
# berechnet die Prognose für das ganze training Set
y: np.ndarray = np.sign(np.dot(w, X.T) + b) # hier ist y ein Vorhersagenvektor

#Q11: Implement the batch version of the Perceptron algorithm.
# Train it on the linear.csv dataset using different parameters (vary the learning rate and weight initialization).
# Compare its performance to the online version.
print(t)
T: TypeVar = TypeVar("T", np.ndarray, List)
def perceptronBatchVersion(X: np.ndarray, t: T, w: T, b: float, eta: float = 0.01, nb_epochs: int = 100) -> Tuple:
    #parameters eta, nb_epochs
    #Gewichtsvektor und Bias -> w/b
    # batch: Perceptron Algorithmus
    errors: List = []
    losses: List = []
    for epoch in range(nb_epochs):
        # berechnet die Prognose für das ganze training Set
        y: np.ndarray = np.sign(np.dot(w, X.T) + b)  # hier ist y ein Vorhersagenvektor
        # Update: Gewichte aktualisieren
        w += eta * np.dot((t - y), X) / N
        # Update: Gewichte aktualisieren
        b += eta * np.sum(t - y) / N
        # berechnet error and loss
        error = np.mean(t != y)
        loss = np.mean((t - y) ** 2)

        #append

        errors.append(error)
        losses.append(loss)

    return losses, errors, w, b

losses, erros, w, b = perceptronBatchVersion(X,t,w,b)
print(losses)
visualize(X,t,w,b, title= "Perceptron Online")

plt.plot(losses, label= "losses")
plt.plot(errors, label= "errors")
plt.legend()
plt.show()











