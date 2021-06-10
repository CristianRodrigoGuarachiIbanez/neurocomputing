import MNIST as mnist
from pooling_layer import MaxPooling
from conv_layer import ConvolutionalLayer
from softmax_layer import Softmax
import numpy as np

######################## Training Set###############################
# Das mnist-Paket verwaltet den MNIST-Datensatz
# # Mehr davon hier https://github.com/datapythonista/mnist
train_images:np.array = mnist.train_images()
train_labels:np.array = mnist.train_labels()

conv = ConvolutionalLayer(8)                  # 28x28x1 -> 26x26x8
pool = MaxPooling()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)    #  13x13x8 -> 10

output = conv.forward_pass(train_images[0])
print(output)# (26, 26, 8)
output2 = pool.forward_pass(output)
print(output2.shape) # (13, 13, 8)

###########################Test Set ####################################
# Hierbei wird nur die ersten 1k Test-Beispiele  (von dem gesamten 10k )
# Um Zeit zu sparen. könnte man dies verändern.
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]


def forward(image: np.array, label: int) -> tuple:
 
  # Das Bild wird hier von [0, 255] auf [-0.5, 0.5] umgewandelt, um es einfacher zu machen
  # Das ist angeblich: "standard practice"
  output: np.array = conv.forward_pass((image / 255) - 0.5)
  output: np.array = pool.forward_pass(output)
  output: np.array = softmax.forward_pass(output)

  # wird cross-entropy loss and accuracy berechnet. np.log() ist das natürliche Log.
  losses: np.array = -np.log(output[label])
  accuracy: int = 1 if np.argmax(output) == label else 0

  return output, losses, accuracy

print('Das Convolutional-Neurale Netzwerk  wurde initializiert!')

loss: int = 0
correct_pred: int = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):

  # Forward-Pass

  _, l, accuracy = forward(im, label)
  loss += l
  correct_pred += accuracy

  # jede 100 Schritte Loss funktion und Accuracy ausgeben lasssen

  if i % 100 == 99:
    print(
      '[Schritt %d] nach 100 Schritten: Mittelwert Loss %.3f | Accuracy: %d%%' %
      (i + 1, loss / 100, correct_pred)
    )
    loss: int = 0
    correct_pred: int = 0
