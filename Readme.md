License Plate Recognition using RCNN + YOLOv8

This project implements a full license plate recognition pipeline:

YOLOv8 is used for plate detection

A Recurrent Convolutional Neural Network (RCNN) with CTC loss is used for text recognition

The system detects license plates in images and then extracts the plate number from each detected region of interest (ROI).


0) Dataset

This project uses the UC3M-LP dataset:

https://github.com/ramajoballester/UC3M-LP

The dataset is particularly convenient because it provides:

Full plate labels (lp_id)

Image dimensions

Polygon coordinates of each plate

Bounding boxes for individual characters

Although character-level annotations are available, this project uses a sequence-based RCNN model and therefore does not rely on per-character bounding boxes.

Special thanks to the dataset authors — without this dataset, manual annotation would have taken many hours.

1) Preparing Data for YOLOv8

To retrain YOLOv8 for plate detection, .json annotations were converted into YOLO-format .txt files.

From each JSON file:

imageWidth and imageHeight were extracted

poly_coord was converted into a bounding rectangle

The bounding box was normalized to YOLO format:

<class_id> <x_center> <y_center> <width> <height>

All coordinates were normalized relative to image width and height.

2) Preparing Data for OCR (RCNN)

For OCR training, a custom dataloader format was created.

Each sample is stored as:

<absolute_path_to_image> <plate_text>

Example:

C:/.../images/train/00001.jpg AN597LK

The label string is cleaned to include only:

A–Z

0–9

This ensures compatibility with CTC loss and a fixed vocabulary of 36 characters plus one blank token.

3.) Model architechture
Model consists of 

CNN backbone

Extracts visual features from the plate image

Bidirectional LSTM

Models sequential dependencies across the plate width in both directions

Linear layer

Maps features to character logits

CTC loss

Handles alignment between predicted sequences and ground truth labels

The model does not require explicit character segmentation.

4) Training Strategy

Images are resized to (64, 256)

Batch size: 32

Optimizer: Adam

Loss: nn.CTCLoss(blank=0)

Validation loss is monitored to detect overfitting.

5) Inference Pipeline

Detect plates using YOLOv8

Crop plate ROI

Resize and normalize image

Run RCNN forward pass

Apply greedy CTC decoding

Output final plate text

6.) Images


Notes

Dataset is not included in this repository.

Please download it from the official source.

Model's weight are published

This project demonstrates an end-to-end applied computer vision pipeline combining object detection and sequence-based OCR.


