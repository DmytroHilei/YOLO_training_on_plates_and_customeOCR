from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch

def code():
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    char_to_idx = {c: i+1 for i, c in enumerate(chars)}
    idx_to_char = {i+1: c for i, c in enumerate(chars)}
    return char_to_idx, idx_to_char


class PlateDataset(Dataset):
    def __init__(self, label_file):
        self.samples = []

        with open(label_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, label))

        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
        ]) # transform on the images. Resizing and making them tensor

    def __len__(self):
        return len(self.samples)

    def encode(self, text):
        char_to_idx, _ = code()
        return torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        encoded_label = self.encode(label)

        return image, encoded_label

#Collecting images, targets from batch
def collate_fn(batch):
    images = []
    targets = []
    target_lengths = []

    for image, label in batch:
        images.append(image)
        targets.append(label)
        target_lengths.append(len(label))

    images = torch.stack(images)
    targets = torch.cat(targets)

    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths
