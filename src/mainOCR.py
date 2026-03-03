import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from GetTextEasyOCR import RCNN, train_one_epoch, validate
from torch.utils.data import DataLoader
from dataloader import PlateDataset, collate_fn

num_epochs = 45
num_classes = 37

train = PlateDataset(r"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\labelsForOCR\train.txt")
val = PlateDataset(r"C:\Users\giley\PycharmProjects\Plates_YOLO_Training\labelsForOCR\val.txt")

train_loader = DataLoader(
    train,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RCNN(num_classes).to(device)

criterion = nn.CTCLoss(blank=0, reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )
    val_loss = validate(
        model,
        val_loader,
        criterion,
        device
    )
    print(f"Epoch {epoch + 1}: {train_loss:.4f}")

torch.save(model.state_dict(), 'model.pth')

