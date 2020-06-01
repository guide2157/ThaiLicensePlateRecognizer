import cv2
import numpy as np


def crop(img, boxes, idx):
    (xtop, ytop) = (boxes[idx][0], boxes[idx][1])
    (w, h) = (boxes[idx][2], boxes[idx][3])
    firstCrop = img[ytop:ytop + h, xtop:xtop + w]
    return cv2.resize(firstCrop, (300, 100))


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def character_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
    edges = auto_canny(thresh_inv)

    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = img.shape[0] * img.shape[1]
    charList = []

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w * h
        non_max_sup = roi_area / img_area

        if (non_max_sup >= 0.015) and (non_max_sup < 0.09):
            if (h > w * 0.85) and (3 * w >= h):
                char = img[y:y + h, x:x + w]
                resized = cv2.resize(char, (50, 45))
                charList.append(resized)
    return charList
