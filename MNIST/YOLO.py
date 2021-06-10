import cv2
#from cv2.dnn import
import numpy as np
from numpy import ndarray
from typing import List, Tuple, Any, Callable

# Load Yolo-Algorithm
net: Any = cv2.dnn.readNet("yolov3.weights", "darknet/cfg/yolov3.cfg")
classes: List[str] = []
with open("darknet/data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(len(classes))
layer_names: List[str] = net.getLayerNames() # 254
output_layers: List[str] = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # 3
print(len(layer_names))
colors: ndarray = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img: Callable = cv2.imread("IMG_0277.JPG")
img: ndarray = cv2.resize(img, None, fx=0.2, fy=0.2)

height, width, channels = img.shape
# Detecting objects
"""Blob it’s used to extract feature from the image and to resize them. YOLO accepts three sizes:"""
blob: Callable = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
# for b in blob:
#     for n, img_blog in enumerate(b):
#         cv2.imshow(f"{n}", img_blog)
# ----------- implementation of the pass fordware ---------
net.setInput(blob)
outs: ndarray = net.forward(output_layers)
#print(outs)

# Showing informations on the screen
class_ids: List = []
confidences: List = []
boxes: List = []
for out in outs:
    for detection in out:
        #print(len(detection))
        # cacualte the confidence
        scores: ndarray = detection[5:]
        #print(len(scores))
        class_id: ndarray = np.argmax(scores)
        #print(class_id)
        confidence: ndarray = scores[class_id]
        #print(confidence)

        if confidence > 0.5:
            # Object detected
            center_x: int = int(detection[0] * width)
            center_y: int = int(detection[1] * height)
            w: int = int(detection[2] * width)
            h: int = int(detection[3] * height)

            # Rectangle coordinates
            x: int = int(center_x - w / 2)
            y: int = int(center_y - h / 2)
            #cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print('detected objects:', len(boxes)) # this gives the number of detected objects
detected_objects: int = len(boxes)

# Non Maximal S
indexes: ndarray = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes.flatten())
font: int = cv2.FONT_ITALIC

for object in range(detected_objects):
    if object in indexes:

        x,y,w,h = boxes[object] # type: int, int, int, int
        label: str = str(classes[class_ids[object]])
        confidence: str = str(round(confidences[object],2))
        color: int = colors[object]
        cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label + " " + confidence, (x, y + 30), font, 1, (0,0,0), 3 )



cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()