import torch
import torch.nn as nn
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

@torch.no_grad()
def countDeadNeurons(
    sae: nn.Module,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer: int = -1,
    device: torch.device | str | None = None,
) -> tuple[int, torch.Tensor]:

    sae.eval()
    model.eval()

    device = device or next(sae.parameters()).device
    sae.to(device)
    model.to(device)

    num_features = sae.num_features
    seen_nonzero = torch.zeros(num_features, dtype=torch.bool, device=device)

    for batch in dataloader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.to(device)

        feats = model(pixel_values=imgs).hidden_states[layer][:, 0]
        _, z, _ = sae(feats)                    

        seen_nonzero |= (z != 0).any(dim=0)

    dead_mask = ~seen_nonzero
    dead_count = int(dead_mask.sum().item())
    return dead_count, dead_mask.cpu()