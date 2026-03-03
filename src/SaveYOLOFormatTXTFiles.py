import json
import os

def convert_json_to_yolo(json_path, output_txt_path, class_id=0):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    yolo_lines = []

    for lp in data["lps"]:
        poly = lp["poly_coord"]

        xs = [point[0] for point in poly]
        ys = [point[1] for point in poly]

        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)

        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(
            f"{class_id} {x_center} {y_center} {width} {height}"
        )

    with open(output_txt_path, 'w') as f:
        for line in yolo_lines:
            f.write(line + "\n")


json_folder = r"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\UC3M-LP"
output_folder = r"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\UC3M-LP\labels_val"

for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(json_folder, filename)
        txt_name = filename.replace(".json", ".txt")
        output_txt_path = os.path.join(output_folder, txt_name)

        convert_json_to_yolo(json_path, output_txt_path)