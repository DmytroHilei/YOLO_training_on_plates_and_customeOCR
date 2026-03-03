import time
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path
from GetTextFromRoi import ocr_tesseract_plate
import re

model = YOLO(R"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\runs\detect\train12\weights\best.pt")

test_dir = Path(R"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\testOCRDataset")


pattern = r'^[A-Z]{2}\d{4}[A-Z0-9]$'

def checkPlate(text):
    return bool(re.match(pattern, text))

conf_every = []
for file in test_dir.glob("*.jpg"):
    img = cv2.imread(str(file))
    if img is None:
        continue

    results = model(img, conf=0.5)
    all_boxes = []
    all_confidences = []

    r = results[0]
    confs = r.boxes.conf.cpu().numpy()
    boxes = r.boxes.xyxy.cpu().numpy()

    all_boxes.extend(boxes)
    all_confidences.extend(confs)
    conf_every.extend(confs)

    h, w = img.shape[:2]

    for (x1, y1, x2, y2), conf in zip(all_boxes, all_confidences):
        #cv2.rectangle(
            #img,
            #(int(x1), int(y1)),
            #(int(x2), int(y2)),
            #(0, 255, 0),
            #2
        #)
        pad = 10
        x1 = max(0, int(x1 - pad))
        y1 = max(0, int(y1 - pad))
        x2 = min(w, int(x2 + pad))
        y2 = min(h, int(y2 + pad))

        roi = img[y1:y2, x1:x2]
        roi = cv2.resize(roi, (320, 80))
        text, _ = ocr_tesseract_plate(roi)
        if checkPlate(text):
            with open("plates.txt", "a") as f:
                f.write(text + "\n")
        print(text)
        print("\n")
    #cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(np.mean(conf_every))


