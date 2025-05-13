
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

class ClipImageDataset(Dataset):

    def __init__(self, folder_path: str, processor: CLIPProcessor):
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
        return pixel_values.squeeze(0)


def createImageDataset(
    path: str,
    model_name: str = "openai/clip-vit-base-patch32"
) -> ClipImageDataset:

    processor = CLIPProcessor.from_pretrained(model_name)
    return ClipImageDataset(path, processor)


def createImageDataloader(
    path: str,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
    shuffle: bool = True
) -> DataLoader:

    dataset = createImageDataset(path, model_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )
