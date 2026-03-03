import torch
import torch.nn as nn

class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()

        # Convolutional feature extractor
        #
        # This CNN block extracts hierarchical visual features from input images.
        #
        # Architecture:
        # - Conv(3 → 64) + ReLU
        # - Conv(64 → 64) + ReLU
        # - MaxPool (downsampling by factor of 2)
        # - Conv(64 → 128) + ReLU
        # - MaxPool (downsampling by factor of 2)
        #
        # Details:
        # - Kernel size = 3x3 with padding=1 preserves spatial dimensions.
        # - ReLU introduces non-linearity.
        # - MaxPooling reduces spatial resolution and increases receptive field.
        #
        # Output:
        # Produces 128 feature maps with spatial size reduced by 4×
        # (due to two MaxPool layers).
        #
        # Purpose:
        # To transform raw RGB images into compact, high-level feature
        # representations suitable for sequence modeling (e.g., RCNN / LSTM).

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # input: 128-dim feature vector (from Sequential)
        #2 layers, 255 hidden units per direction
        # Left to right and right to left context matters

        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
        )
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()

        x = x.mean(2)
        x = x.permute(2, 0, 1)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x.log_softmax(dim=2)



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, targets, target_lengths in dataloader:

        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        log_probs = model(images)

        T, batch_size, _ = log_probs.size()

        input_lenghts = torch.full(
                size=(batch_size,),
                fill_value=T,
                dtype=torch.long,
        ).to(device)

        loss = criterion(
                log_probs,
                targets,
                input_lenghts,
                target_lengths,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(images)

            T, batch_size, _ = log_probs.size()

            input_lenghts = torch.full(
                size=(batch_size,),
                fill_value=T,
                dtype=torch.long,
            ).to(device)

            loss = criterion(
                log_probs,
                targets,
                input_lenghts,
                target_lengths,
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)
