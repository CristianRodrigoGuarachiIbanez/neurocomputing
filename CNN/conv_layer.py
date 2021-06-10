import numpy as np
from typing import Tuple, List, Dict

class ConvolutionalLayer:

  def __init__(self, num_filters:int):

    # das setzt dia Anzahl an Filtern, die man gebraucht
    self.num_filters:int = num_filters

    # Hier wird die Filtern erstellt (3 x 3)
    # Der Filter ist ein 3d-Array mit den folgenden Dimensionen (num_filters, 3, 3)
    # Angeblich musste man hier die Varianz der Ausgangswerte reduzieren, indem man durch 9 teilt
    self.filters: np.array = np.random.randn(num_filters, 3, 3) / 9

  def iterate_regions(self, image: np.array) -> Tuple[np.ndarray, int, int]:
      """ iterate über das ganze Bild, über jedes Pixel
      @:parameter np.array from Image
      @:return tuple mit dem Bildbereich, und das Verzeichnis des Bildpixels in x und y achsen"""
     
      h,w = image.shape # type: int, int

      #wird durch das Bild von Recht nach Links, von Oben nach unten iteriert
      #ein neuer Bildbereich plus die Koordinaten werden nach jeder Iteation zurückgegeben
      for i in range(h - 2):
          for j in range(w - 2):
              # der Bildbereich ist dann 3x3 und Schrittweise (Stride) bewegt das sich um 1, daher i:(i * 3)
              im_region: np.array = image[i:(i + 3), j:(j + 3)]
              yield im_region, i, j

  def forward_pass(self, input: np.array) -> np.array: # as float zahl
      """
    @:param input: ein np.array  von Bildpixeln
    @:return: liefert ein Array züruck, welcher nur dem ausgewählten Bildbereich entspricht
    """
      h, w = input.shape #type: int, int
      # (Anzahl der Filter, Zeile, Spalten)
      output: np.array = np.zeros((h - 2, w - 2, self.num_filters))

      # man erhält für jede Iteration einen neuen Bildbereich plus ihre Koordiaten
      for im_region, i, j in self.iterate_regions(input):
          #Anhand der Koordinaten wird das neue Ergebnis sum (np.dot(bildbreich, filtern)) in das Output positionert
          output[i, j] = np.sum(im_region @ self.filters, axis=(1, 2)) # im_region: (1,3) @ (3,3) = (1,3)

      return output
