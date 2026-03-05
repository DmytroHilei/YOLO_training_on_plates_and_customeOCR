import torch
import torch.nn as nn
import torch.optim as optim
from CreateArchitectureOfModel import RCNN, train_one_epoch, validate
from torch.utils.data import DataLoader
from dataloader import PlateDataset, collate_fn

#choose number of epochs. 45 could be too small
num_epochs = 45
#A - Z letter + 1 + digits = 37 classes, no special symbols
num_classes = 37

#upload dataset

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
#Choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Create model
model = RCNN(num_classes).to(device)

#loss function
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

#Save only weights
torch.save(model.state_dict(), 'model.pth')

