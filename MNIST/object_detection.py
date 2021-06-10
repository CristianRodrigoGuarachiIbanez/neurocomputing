import cv2
#from cv2.dnn import
import numpy as np
from numpy import ndarray
from typing import List, Tuple, Any, Callable, Iterable, Optional, TypeVar

class YOLO_Algorithm:
    def __init__(self, model: Any = "yolov3.weights", conf: str = "darknet/cfg/yolov3.cfg" ):

        self._net: Any = cv2.dnn.readNet(model, conf)
        self._classes: List[str] = self.__getClasses()

        self._layer_names: List[str] = self._net.getLayerNames()  # 254
        print('NUMBER OF LAYERS:',len(self._layer_names))
        self._output_layers: List[str] = [self._layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]  # 3
        print('NUMBER OF OUTPUT LAYERS',len(self._output_layers))
        self._colors: ndarray = np.random.uniform(0, 255, size=(len(self._classes), 3))

    def __getClasses(self) -> List[str]:
        classes: List[str] = []
        with open("darknet/data/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print('NUMBER OF CATEGORIES IN OUTPUT LAYER',len(classes))
        return classes



class ObjectDetector(YOLO_Algorithm):

    T = TypeVar('T', List[float], str)

    def __init__(self, path: str) -> None:
        super(ObjectDetector, self).__init__()
        img: Callable = cv2.imread(path)
        self.__img: ndarray = cv2.resize(img, None, fx=0.25, fy=0.25)
        self.__height, self.__width, self.__channels = self.__img.shape # type: int, int, int
        #detecting object
        self.__output: ndarray = self.__passforeward()

        # show Image
        #self.__showImg()
    def getObjectDetected(self) -> None:
        font: int = cv2.FONT_ITALIC
        for object in self.__applyNonMaxSupression().flatten():
            x, y, w, h = self.__getDatafromOutputLayer()[0][object]  # type: int, int, int, int
            label: str = str(self._classes[self.__getDatafromOutputLayer()[2][object]])
            confidence: str = str(round(self.__getDatafromOutputLayer()[1][object], 2))
            color: int = self._colors[object]
            cv2.rectangle(self.__img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(self.__img, label + " " + confidence, (x, y + 30), font, 1, (0, 0, 0), 3)

    def __applyNonMaxSupression(self) -> ndarray:
        return cv2.dnn.NMSBoxes(self.__getDatafromOutputLayer()[0], self.__getDatafromOutputLayer()[1], 0.5, 0.4)

    def getImgWithBoundingBox(self, confidence: Optional[bool] = None, boundingBoxes: Optional[bool] = True, class_id: Optional[bool] = None ) -> T:
        """:parameter recives a boolean parameter true or false
        .:return a List with the confidence values or the categories, or a number of detected object on the img"""
        if(confidence):
            return self.__getDatafromOutputLayer()[1]
        elif(boundingBoxes):
            return f'NUMBER OF DETECTED OBJECTS {len(self.__getDatafromOutputLayer()[0])}'
        elif(class_id):
            return self.__getDatafromOutputLayer()[2]

    def __getDatafromOutputLayer(self) -> Tuple[List, List, List]:
        """search for detections at the output layer
        @:return a tuple containing the list of confidence """
        class_ids: List = []
        confidences: List = []
        boxes: List = []
        for out in self.__output:
            for detection in out:
                # cacualte the confidence
                scores: ndarray = detection[5:]
                # print(type(scores))
                class_id: ndarray = np.argmax(scores) # get the index of the max value in the list
                # print(class_id)
                confidence: ndarray = scores[class_id] # uses the given index und give the value item aout of the list (score)
                # print(confidence)

                if confidence > 0.5:
                    # Object detected
                    center_x: int = int(detection[0] * self.__width)
                    center_y: int = int(detection[1] * self.__height)
                    w: int = int(detection[2] * self.__width)
                    h: int = int(detection[3] * self.__height)

                    # Rectangle coordinates
                    x: int = int(center_x - w / 2)
                    y: int = int(center_y - h / 2)
                    #cv2.rectangle(self.__img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes, confidences, class_ids


    def __passforeward(self, showBlob: Optional[bool] = None) -> ndarray:
        blob: Iterable = cv2.dnn.blobFromImage(self.__img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        if(showBlob):
            self.__showBlobImg(blob)
        # ----------- implementation of the pass fordware ---------
        self._net.setInput(blob)
        return self._net.forward(self._output_layers)
    @staticmethod
    def __showBlobImg(blob: Iterable) -> None:
        for b in blob:
            for n, img_blog in enumerate(b):
                cv2.imshow(f"{n}", img_blog)

    def showImg(self) -> None:
        cv2.imshow("Image", self.__img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == '__main__':
    object_detector: ObjectDetector = ObjectDetector('IMG_1679.JPG')
    print(object_detector.getImgWithBoundingBox())
    object_detector.getObjectDetected()
    object_detector.showImg()


