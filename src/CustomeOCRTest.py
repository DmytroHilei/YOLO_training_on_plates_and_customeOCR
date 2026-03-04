import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from CreateArchitectureOfModel import RCNN
from ModelTest import findRoi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 37
model = RCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("../models/model.pth", map_location=device))
model.to(device)
model.eval()



transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
])
img = cv2.imread(r"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\UC3M-LP\test\00022.jpg")
if img is None:
    print("Could not load the image")
    exit(1)

roi = findRoi(img)

roi_pil = Image.fromarray(roi[:, :, ::-1])
roi_tensor = transform(roi_pil).unsqueeze(0).to(device)
with torch.no_grad():
    log_probs = model(roi_tensor)

chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
idx_to_char = {i+1: char for i, char in enumerate(chars)}

def decode(log_probs):
    preds = torch.argmax(log_probs, dim=2)
    preds = preds.permute(1, 0)

    decoded_texts = []

    for pred in preds:
        prev = -1
        text = ""
        for p in pred:
            p = p.item()
            if p != prev and p != 0:
                text += idx_to_char[p]
            prev = p
        decoded_texts.append(text)

    return decoded_texts

result = decode(log_probs)[0]
cv2.imshow("photo", roi)
print(result)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    exit(0)
