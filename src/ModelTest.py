import time
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path
#from GetTextFromRoi import ocr_tesseract_plate
import re

model = YOLO(R"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\runs\detect\train12\weights\best.pt")

test_dir = Path(R"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\testOCRDataset")


pattern = r'^[A-Z]{2}\d{4}[A-Z0-9]$'

def checkPlate(text):
    return bool(re.match(pattern, text))
def findRoi(img):

    results = model(img, conf=0.5)
    r = results[0]

    boxes = r.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0]

    h, w = img.shape[:2]
    pad = 10

    x1 = max(0, int(x1 - pad))
    y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad))
    y2 = min(h, int(y2 + pad))

    roi = img[y1:y2, x1:x2]
    roi = cv2.resize(roi, (320, 80))

    return roi