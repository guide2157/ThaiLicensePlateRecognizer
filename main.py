import argparse
from pathlib import Path
import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt

from CNNRecognizer.CharacterRecognizer import CharRecognizer
from Yolov3PlateDetection.CharactersExtractor import character_detection
from Yolov3PlateDetection.PlateExtractor import PlateExtractor


def execute_models(yolo_weight, yolo_cfg, cnn_weight, label_dir, img_dir, verbose=True):
    extractor = PlateExtractor(yolo_weight, yolo_cfg)
    recognizer = CharRecognizer(cnn_weight)
    recognizer.load_label(label_dir)

    imgs_dir = []
    if os.path.isdir(img_dir):
        for path in Path(img_dir).glob('.jpg'):
            imgs_dir.append(path)
    else:
        imgs_dir.append(img_dir)

    if len(imgs_dir) == 0:
        print("No image file detected.")
        exit()

    crops = np.zeros((len(imgs_dir), 100, 300, 3), dtype=np.uint8)
    if verbose:
        print("Extracting license plates...\n")
    for i in tqdm.tqdm(range(len(imgs_dir))):
        crops[i] = extractor.extract_plate(imgs_dir[i])

    if len(crops) == 0:
        exit(print("No license detected."))

    license_list = []
    if verbose:
        print("Extracting characters from plates...\n")
    for cropped in crops:
        license_list.append(character_detection(cropped))

    results = []
    if verbose:
        print("Recognizing characters...\n")
    for char_list in license_list:
        results.append(recognizer.predict(np.array(char_list)))

    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='THLicenseReg', description='Thai License Plate Recognizer')
    parser.add_argument('-v', '--verbose', type=bool, required=False)
    parser.add_argument('yolo_weight_dir', type=str, help='Directory of weight file from trained YOLOv3 model')
    parser.add_argument('yolo_cfg_dir', type=str, help='Directory of configuration file from YOLOv3 model')
    parser.add_argument('cnn_dir', type=str, help='Directory of weight file from trained CNN model')
    parser.add_argument('label_dir', type=str, help='Directory of dictionary for labels')
    parser.add_argument('img_dir', type=str, help='Directory of the image')

    args = parser.parse_args()

    execute_models(args.yolo_weight_dir, args.yolo_cfg_dir, args.cnn_dir, args.label_dir, args.img_dir, args.verbose)
