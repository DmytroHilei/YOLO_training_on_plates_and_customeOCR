import time

import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO(R"C:\Users\giley\PycharmProjects\YOLO_training_on_plates_and_customeOCR\models\best.pt")

img = cv2.imread(R"C:\Users\giley\Downloads\parking_car.jpg")

if img is None:
    raise ValueError("Cannot read the image")
result = model(img, conf=0.1, imgsz=960, verbose=False)
r = result[0]

if r.boxes is None:
    raise ValueError("Cannot detect car")

boxes = r.boxes.xyxy.cpu().numpy()
confs = r.boxes.conf.cpu().numpy()
if len(boxes) == 0:
    raise ValueError("Cannot detect car")

print("conf min/mean/max:", confs.min(), confs.mean(), confs.max())
print("first box:", boxes[0])
for (x1, y1, x2, y2), c in zip(boxes, confs):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"{c:.2f}", (int(x1), max(0, int(y1)-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

annotated = result[0].plot()
cv2.imshow("img", annotated)
cv2.waitKey(0)
cv2.imshow("img", img)
if cv2.waitKey(0) & 0xff == ord('q'):
    cv2.destroyAllWindows()
    exit(0)
