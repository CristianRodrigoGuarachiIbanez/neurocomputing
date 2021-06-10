import cv2
from cv2 import VideoCapture
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


class recogImgOnVideo(YOLO_Algorithm):
    T = TypeVar('T', List[float], str)

    def __init__(self, path: int = None) -> None:
        super(recogImgOnVideo, self).__init__()

        #img: Callable = cv2.imread(path)
        self.__cap: VideoCapture = VideoCapture(path)

        # Check if the webcam is opened correctly
        if not self.__cap.isOpened():
            raise IOError("Cannot open webcam")
        self.__mainLoop()

    def __mainLoop(self) -> None:
        """
        run the main loop
        """

        while True:
            _, _img = self.__cap.read() # type: None, Any

            self.__img: ndarray = cv2.resize(_img, (725,540), fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
            self.__height, self.__width, self.__channels = self.__img.shape  # type: int, int, int

            # Detecting objects
            self.__output: ndarray = self.__passforeward()

            # Showing informations on the screen
            self.__detectingObjects()
            key: int = self.__showImg()
            if(key == 27):
                break


    def __detectingObjects(self) -> None:
        """
        iterate over the detected Bboxes, but just the ones which are not supresed will be shown
        """

        __boxes, __confidences, __class_ids = self.__getDatafromOutputLayer(self.__output, self.__height, self.__width,
                                                                             self.__channels)  # type: List, List, List
        detected_objects: int = len(__boxes)

        # Non Maximal Supression
        indexes: ndarray = cv2.dnn.NMSBoxes(__boxes, __confidences, 0.5, 0.4)
        font: int = cv2.FONT_ITALIC

        for object in range(detected_objects):
            if object in indexes:
                x, y, w, h = __boxes[object]  # type: int, int, int, int
                label: str = str(self._classes[__class_ids[object]])
                confidence: str = str(round(__confidences[object], 2))
                color: int = self._colors[object]
                cv2.rectangle(self.__img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.__img, label + " " + confidence, (x, y + 30), font, 1, (0, 0, 0), 3)


    @staticmethod
    def __getDatafromOutputLayer(__output: ndarray, __height: int, __width: int, __channels: int) -> Tuple[List, List, List]:
        """search for detections at the output layer
        @:return a tuple containing the list of confidence """
        class_ids: List = []
        confidences: List = []
        boxes: List = []
        for out in __output:
            for detection in out:
                # calculate the confidence
                scores: ndarray = detection[5:]
                # print(type(scores))
                class_id: ndarray = np.argmax(scores)  # get the index of the max value in the list
                # print(class_id)
                confidence: ndarray = scores[class_id]  # uses the given index und give the value item aout of the list (score)
                # print(confidence)

                if confidence > 0.5:
                    # Object detected
                    center_x: int = int(detection[0] * __width)
                    center_y: int = int(detection[1] * __height)
                    w: int = int(detection[2] * __width)
                    h: int = int(detection[3] * __height)

                    # Rectangle coordinates
                    x: int = int(center_x - w / 2)
                    y: int = int(center_y - h / 2)
                    # cv2.rectangle(self.__img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes, confidences, class_ids

    def __passforeward(self, showBlob: Optional[bool] = None) -> ndarray:
        """
        :param showBlob: a boolean operator
        :return: a numpy array where the outputs the CNN are contained
        """
        blob: Iterable = cv2.dnn.blobFromImage(self.__img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        if (showBlob):
            self.__showBlobImg(blob)

        # ----------- implementation of the pass fordward ---------
        self._net.setInput(blob)
        return self._net.forward(self._output_layers)

    @staticmethod
    def __showBlobImg(blob: Iterable) -> None:
        """

        :param blob:
        :return:
        """
        for b in blob:
            for n, img_blog in enumerate(b):
                cv2.imshow(f"{n}", img_blog)

    def __showImg(self) -> int:
        cv2.imshow("Image", self.__img)
        return cv2.waitKey(1)


    def close(self) -> None:
        self.__cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    video: recogImgOnVideo = recogImgOnVideo(0)
    video.close()
