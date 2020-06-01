import cv2
import numpy as np

from Yolov3PlateDetection.CharactersExtractor import crop


class PlateExtractor:

    def __init__(self, weights_path, config_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path,weights_path)

    def extract_plate(self, img_dir, threshold=0.2):
        try:
            img = cv2.imread(img_dir)
        except AttributeError:
            raise Exception("Cannot find the image directory.")
        (H, W) = img.shape[:2]

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False)

        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
        if len(idxs) > 0:
            cropped = crop(img, boxes, idxs[0][0])
            return cropped
        return None
