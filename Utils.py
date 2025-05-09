from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class FlatImageDataset(Dataset):
    def __init__(self, folder_path, resizew, resizeh, transform=None):
        self.folder_path = folder_path
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform or transforms.Compose([
            transforms.Resize((resizew, resizeh), interpolation=Image.BICUBIC),
            transforms.CenterCrop((resizew, resizeh)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img).flatten()
        return img, 0 # Dummy Label


def createImageDataset(path, resizew, resizeh):
    return FlatImageDataset(path, resizew, resizeh)


def createImageDataloader(path, resizew, resizeh, bsize=64, shuffle=True):
    dataset = createImageDataset(path, resizew, resizeh)
    dataloader = DataLoader(dataset, batch_size=bsize, shuffle=shuffle, num_workers=2)
    return dataloader