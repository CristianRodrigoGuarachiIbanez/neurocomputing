import numpy as np
from typing import List, Tuple
class MaxPooling:


  def iterate_regions(self, image: np.array) -> Tuple[np.ndarray, int, int]:
    """

    :param image:
    :return:
    """
   
    h, w, _ = image.shape
    new_h: int = h // 2
    new_w: int = w // 2

    for i in range(new_h):
      for j in range(new_w):
        # der Filter ist dann 2x2, daher i:(i * 2), aber bewegt sich Schrittweise (Stride) um 2,
        # daher (i * 2):(i * 2 + 2)
        im_region: np.array = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def forward_pass(self, input:np.array) -> np.array:
    """

    :param input: array das ist der Bildbereich
    :return: liefert ein Array zurück, der
    """
    
    h, w, num_filters = input.shape # 3D-Numpy-Array
    output: np.array = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j in self.iterate_regions(input):
        # Die Bildbereich wird maximiziert, um eine 1 x 3 Matrix zu erhalten
        output[i, j] = np.amax(im_region, axis=(0, 1))

    return output
