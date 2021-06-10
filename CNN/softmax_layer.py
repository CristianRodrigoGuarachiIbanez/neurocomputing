import numpy as np
from typing import List, Tuple
class Softmax:
  
  def __init__(self, input_len:int, nodes:int) -> None:
    """

    :param input_len:
    :param nodes:
    """

    # wird durch input_len dividiert, um die Varianz unserer Ausgangswerte zu reduzieren

    self.weights: np.array = np.random.randn(input_len, nodes) / input_len
    self.biases: np.array = np.zeros(nodes)

  def forward_pass(self, input: np.array) -> np.array:
    """

    :param input:
    :return:
    """
    #https://www.w3resource.com/numpy/manipulation/ndarray-flatten.php
    input: np.array = input.flatten()

    # die Gewichte werden in Zeile () und Spalte der Matrix getrennt
    #input_len, nodes = self.weights.shape

    # y = W*xi + ß
    totals: np.array = np.dot(input, self.weights) + self.biases
    exp: np.array = np.exp(totals)

    # aL = ti/sum(ti) wobei t = exp(zL) und z = WL * a(L-1) + bL
    return exp / np.sum(exp, axis=0)
