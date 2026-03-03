import json
import os
import re


json_dir = r"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\UC3M-LP\images\val"
output_file = r"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\labelsForOCR\val.txt"
img_path_dir = r"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\UC3M-LP\images\val"



def clean_plate(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

with open(output_file, "w") as out_f:
    for filename in os.listdir(json_dir):
        json_path = os.path.join(json_dir, filename)
        print("Opening:", json_path)
        if filename.endswith(".json"):
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        else:
            continue
        if len(json_data["lps"]) == 0:
            continue

        label = json_data["lps"][0]["lp_id"]
        label = clean_plate(label)

        img_name = filename.replace(".json", ".jpg")

        img_path = fr"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\UC3M-LP\images\val\{img_name}"
        out_f.write(f"{img_path} {label}\n")
