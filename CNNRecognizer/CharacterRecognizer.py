from tensorflow.keras import models
import numpy as np
import pickle


class CharRecognizer:
    def __init__(self, weight_dir):
        self.model = models.load_model(weight_dir)
        self.labels = None

    def load_label(self, label_dir):
        with open(label_dir, 'rb') as handle:
            le_dict = pickle.load(handle)
            self.labels = {v: k for k, v in le_dict.items()}

    def translate_prediction(self, prediction):
        license_number = ''
        for pred in prediction:
            if pred != -1:
                license_number += self.labels[pred]
        return license_number

    def predict(self, img, threshold=0.2):
        assert img[0].shape == (45, 50, 3)
        if not self.labels:
            raise Exception("Labels are not loaded into the model. Please call load_label before predict.")
        pred = self.model.predict(img)
        char = np.argmax(pred, axis=1)
        confidence = np.max(pred, axis=1)
        remove_idx = np.argwhere(confidence < threshold)
        char[remove_idx] = -1
        return self.translate_prediction(char)
