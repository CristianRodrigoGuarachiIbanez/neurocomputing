	#Exercise 6 : MNIST classification using keras

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from typing import Tuple, List, Callable, Any, TypeVar

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import to_categorical

#1.1 - Data preprocessing

(X_train, y_train), (X_test, y_test) = mnist.load_data() # type: Tuple, Tuple

# Q3: Print the shape of the four numpy arrays (X_train, y_train), (X_test, y_test) and visualize some training examples to better understand what you are going to work on.

print("Training data set:", X_train.shape, y_train.shape)
print("Test data set:", X_test.shape, y_test.shape)

idx:int = 682
x: np.ndarray = X_train[idx, :] # Features sind die Bildpixeln in form der Ziffer
y: np.ndarray = y_train[idx] # Label oder y var
print("X (shape):", x.shape)
print("y:", y)

plt.imshow(x, cmap="gray")
plt.colorbar()
plt.show()

"""In this exercise, we are going to use a regular MLP (with fully-connected layers).
 Convolutional layers will be seen next time.
REGUALTION
1.- We therefore need to transform the 28x28 input matrix into a 784 vector. 
Additionally, pixel values are integers between 0 and 255. We have to rescale them to floating values in [0, 1]."""

X_train: np.ndarray = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255.
#print("train Shape:" , X_train.shape)
X_test: np.ndarray = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255.

"""
MEAN REMOVAL
2.- We saw in the last exercise that "mean removal" is crucial when training a neural network. 
The following cell removes the mean image of the training set from all examples.
-> mean removal
Standardization or mean removal is a technique that simply centers data by removing the average value of each characteristic, and then scales it by dividing non-constant characteristics by their standard deviation. 
It's usually beneficial to remove the mean from each feature so that it's centered on zero.
"""

X_mean: np.ndarray = np.mean(X_train, axis=0)
print("X MEAN:", X_mean.shape)
X_train -= X_mean #type: np.ndarray
X_test -= X_mean #type: np.ndarray

plt.imshow(X_mean.reshape((28, 28))*255, cmap="gray")
plt.show()

"""
ONE HOT ENCODING TO LABELS
The last preprocessing step is to perform one-hot encoding of the output labels. 
We want for example the digit 4 (index 5 in the outputs y) to be represented by the vector:

[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]

keras offers the utility utils.to_categorical to do that on the whole data:
"""

Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

print(Y_train[idx])

#1.2 - Model definition
def sequetialModels(units: List[int], activation: List[str], input_shape: Tuple,
                    lr: float, metrics: List[str] = ['accuracy'], loss: str = 'categorical_crossentropy') -> Tuple[Sequential, List[float], History]:

    """A sequential model allows us to create models layer by layer in a step by step fashion."""
    model: Sequential = Sequential()

    # Hidden layer with 100 logistic neurons, taking inputs from the 784 pixels
    model.add(Dense(units=units[0], activation=activation[0], input_shape=input_shape))

    # Softmax output layer
    model.add(Dense(units=units[1], activation=activation[1]))

    # Learning rule
    optimizer: SGD = SGD(lr=lr)

    # Loss function
    model.compile(
        loss=loss, # loss function
        optimizer=optimizer, # learning rule
        metrics=metrics # show accuracy
    )

    history: History = History()

    model.fit(
        X_train, Y_train,
        batch_size=100,
        epochs=30,
        validation_split=0.1,
        callbacks=[history]
    )
    score: List = model.evaluate(X_test, Y_test, verbose=0)

    return model.summary(), score, history

activation: List[str] = ['sigmoid', 'softmax']
units: List[int] = [100, 10]
input_shape: Tuple = (784,)
learningRole: float = 0.01
metrics: List[str] = ['accuracy']
#summary, score, history = sequetialModels(units, activation, input_shape, learningRole) # type: Sequential, List[float], History

"""Answer: the hidden layer has a weight matrix of size 784x100 and 100 biases, what makes 78500 free parameters. The output layer has a weight matrix of size 100x10 and 10 biases, so 1010 parameters.
Note that we have more free parameters than training examples, we are going to have to regularize quite hard..."""

#1.3 - Model training

"""Now is time to train the network on MNIST. The following cell creates a History() object that will record the progress of your network.
It then calls the model.fit() method, which tells the network to learn the MNIST dataset defined by the (X_train, Y_train) arrays. You have to specify:

1.- the batch size, i.e. the number of training examples in each minibatch used by SGD.
2.- the maximal number of epochs for training
3.- the size of the validation, taken from the training set to track the progress (this is not the test set!). Here we reserve 10% of the training data to validate. If you do not have much data, you could set it to 0.
4.-  a callback, which will be called at the end of each epoch. Here it will save the metrics defined in model.compile() in the History() object.
The training process can take a while depending on how big your network is and how many data you have. You can interrupt the kernel using the menu if you want to stop the processing in the cell."""

def plottingModel(history: Callable)-> None:
    """You can also use the History() object to visualize the evolution of the the training and validation accuracy during learning.
    Hint: if you are using tensorflow 1.x, replace history.history['accuracy'] and history.history['val_accuracy'] with history.history['acc'] and history.history['val_acc']."""

    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.plot(history.history['loss'], '-r', label="Training")
    plt.plot(history.history['val_loss'], '-b', label="Validation")
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(122)
    plt.plot(history.history['acc'], '-r', label="Training")
    plt.plot(history.history['val_acc'], '-b', label="Validation")
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

"""The training has now run for 20 epochs on the training set. You see the evolution of loss function and accuracy for both the training and validation sets.
To test your trained model on the test set, you can call model.evaluate():

loss - training data, calculated every batch
acc - training data, calculated every batch
val_loss - test data, calculated every epoch
val_acc - test data, calculated every epoch
"""
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])


#plottingModel(history=history)

"""Q5: Did overfitting occur during learning? Why?
Answer: no, the training accuracy is always below the validation accuracy (validation accuracy is higher). The model is too small to overfit.
the other way around: traning set shows more MSE that validation set 
Q6: Looking at the curves, does it make sense to continue learning for much more epochs?
Answer: no, the accuracy have started saturating, it will not get much better (perhaps one percent or two) or very slowly (you can let it learn for 500 epochs or more to see)"""


"""The following cell makes predictions on the test set (`model.predict_classes`) and displays some misclassified examples. Te title of each subplot denotes the predicted class and the groud truth.
Q7: Are some mistakes understandable?"""
from numpy import ndarray

from typing import TypeVar, Any, List, Tuple
from numpy import ndarray


class MyModelCNN(Sequential):
    P = TypeVar("P", SGD, Adam)

    def __init__(self, filter: List[Any], shape: Tuple[int, int, int], maxpoolSize: Tuple[int, int],
                 activation: List[str], units: List[int], lr: List[Any], ConvLayer: bool = False) -> None:
        super(MyModelCNN, self).__init__()
        if (ConvLayer):
            self.__convolutionalLayer: Conv2D = self.add(self.__convLayer(filter=filter[0], kernel_size=filter[1], strides=(1, 1), padding=filter[2],
                                 activation=filter[3], input_shape= shape))
            self.__maxPool: MaxPooling2D = self.add(self.__maxPoolingLayer(pool_size=maxpoolSize))
            self.add(Flatten())

        self.__firstLayer: Dense = self.add(Dense(units=units[0], activation=activation[0]))
        self.__secondLayer: Dense = self.add(Dense(units=units[1], activation=activation[1]))
        self.__thirdLayer: Dense = self.add(Dense(units=units[2], activation=activation[2]))

        self.__compiler(self.__optimazer(lr[0], lr[1], lr[2], lr[3]))
        print(self.__compiler)

    # def __addLayer(self) -> None:
    #     self.add()
    def __maxPoolingLayer(self, pool_size=(2, 2), strides=None, padding="valid", data_format=None) -> MaxPooling2D:
        return MaxPooling2D(pool_size, strides, padding, data_format)

    def __convLayer(self, filter: int, kernel_size: Tuple[int], strides: Tuple[int, int] = (1, 1), padding: str = "valid",
                    activation: str = "relu", input_shape: Tuple[int, int, int] = None) -> Conv2D:
        return Conv2D(filters=filter, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                      input_shape=input_shape)

    def __optimazer(self, name: str, lr: float = 0.1, mo_be: float = 0.0, ne_ams: bool = False) -> P:
        # optimizer: Any = None
        if (name == 'SGD'):
            return SGD(lr=lr, decay=1e-6, momentum=mo_be, nesterov=ne_ams, name=name)
        elif (name == "Adam"):
            return Adam(lr=lr, beta_1=mo_be, beta_2=0.999, epsilon=1e-07, amsgrad=ne_ams, name=name)

    def __compiler(self, optimizer: P) -> None:
        self.compile(
            loss='categorical_crossentropy',  # loss function
            optimizer=optimizer,  # learning rule
            metrics=['accuracy']  # show accuracy
        )

    def HistoryValues(self, X_t: ndarray, Y_t: ndarray, batch_size: int, epochs: int) -> None:
        self.__history: History = History()
        self.__fitting(X_t, Y_t, batch_size, epochs)

    def __fitting(self, X_t: ndarray, Y_t: ndarray, batch_size: int, epochs: int) -> None:

        self.fit(
            X_t, Y_t,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[self.__history]
        )

    def score(self) -> List[float]:
        return self.evaluate(self.__X_t, self.__Y_t, verbose=0)

    def Summary(self) -> Any:
        return self.summary()

    def prediction(self, x: ndarray, plotting: bool = None) -> ndarray:
        # return self.predict_classes(x, verbose=0)
        output: ndarray = self.predict(x, verbose=0)
        prediction: ndarray = output[0].argmax()
        if (plotting):
            plt.figure()
            plt.plot(output[0], 'o')
            plt.xlabel('Digit')
            plt.ylabel('Probability')
        plt.show()
        return prediction

    def parameters(self) -> ndarray:
        return self.layers[0].get_weights()[0]

    def modelOutput(self, inputImg: ndarray, inputLength: int, LayerToBeOutputed: int,
                    plotting: bool = False) -> ndarray:
        model_conv: Model = Model(inputs=self.inputs, outputs=self.layers[LayerToBeOutputed].output)
        features: ndarray = model_conv.predict([x])
        if (plotting):
            plt.figure(figsize=(10, 8))
            for i in range(inputLength):
                plt.subplot(4, 4, i + 1)
                plt.imshow(features[0, :, :, i], cmap=plt.cm.gray, interpolation='nearest')
                plt.xticks([]);
                plt.yticks([])
                plt.colorbar()
            plt.show()
        return features

    def visualizingFilter(self) -> None:
        plt.figure(figsize=(10, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(self.parameters()[:, :, 0, i], cmap=plt.cm.gray, interpolation='nearest')
            plt.xticks([]);
            plt.yticks([])
            plt.colorbar()
        plt.show()

    def visualizing(self, loss: List[str], acc: List[str]) -> None:
        plt.figure(figsize=(15, 6))
        plt.subplot(121)
        plt.plot(self.__history.history[loss[0]], '-r', label="Training")
        plt.plot(self.__history.history[loss[1]], '-b', label="Validation")
        plt.xlabel('Epoch #')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(122)
        plt.plot(self.__history.history[acc[0]], '-r', label="Training")
        plt.plot(self.__history.history[acc[1]], '-b', label="Validation")
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


filter: List[Any] = [16, (3,3), "valid", "relu"]
input_shape: Tuple[int,int,int] = (28,28,1)
maxpooLayer: Tuple[int, int] = (2,2)
unitActivation: List[int] = [100, 50, 10]
netActivation: List[str] = ["sigmoid", "relu", "softmax"]


lr: List[Any] = ['Adam', 0.01, 0.9,True]

mymodel: MyModelCNN = MyModelCNN(filter=filter, activation=netActivation, shape = input_shape, maxpoolSize=maxpooLayer, lr=lr,
                                 units=unitActivation, ConvLayer= True)
mymodel.HistoryValues(X_train, Y_train, 100,5)
#2 - Questions

"Q6: Print the shape of these weights and relate them to the network."

mymodel.parameters().shape
