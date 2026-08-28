import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .preprocessing import compute_ela


class TamperingDataset(Dataset):
    """
    Expects a directory layout:
        root/
          Au/   -> authentic images   (label 0)
          Tp/   -> tampered images    (label 1)
    """

    def __init__(self, root_dir, image_size=128, ela_quality=90):
        self.samples = []
        au_dir = os.path.join(root_dir, "Au")
        tp_dir = os.path.join(root_dir, "Tp")

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        for fname in os.listdir(au_dir):
            if fname.lower().endswith(valid_ext):
                self.samples.append((os.path.join(au_dir, fname), 0))
        for fname in os.listdir(tp_dir):
            if fname.lower().endswith(valid_ext):
                self.samples.append((os.path.join(tp_dir, fname), 1))

        self.ela_quality = ela_quality
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        ela_image = compute_ela(path, quality=self.ela_quality)
        tensor = self.transform(ela_image)
        return tensor, torch.tensor(label, dtype=torch.long)