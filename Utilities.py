import torch
import torch.nn as nn
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import scienceplots
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
        z = sae(feats)[1]

        seen_nonzero |= (z != 0).any(dim=0)

    dead_mask = ~seen_nonzero
    dead_count = int(dead_mask.sum().item())
    return dead_count, dead_mask.cpu()

@torch.no_grad()
def collectLayerData(
    sae: torch.nn.Module,
    vit: torch.nn.Module,
    dataloader,
    layer: int = -1,
    device=None,
):

    sae.eval()
    vit.eval()
    device = device or next(sae.parameters()).device

    imgs, vit_act, sae_z, sae_rec = [], [], [], []

    for batch in dataloader:
        x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)

        imgs.append(x.cpu())

        act = vit(pixel_values=x, output_hidden_states=True)\
                .hidden_states[layer][:, 0]
        vit_act.append(act.cpu())

        _, z, recon = sae(act)[:3]
        sae_z.append(z.cpu())
        sae_rec.append(recon.cpu())

    return dict(
        images     = torch.cat(imgs, dim=0),
        vit_act    = torch.cat(vit_act, dim=0),
        sae_z      = torch.cat(sae_z, dim=0),
        sae_recon  = torch.cat(sae_rec, dim=0),
    )

def findImagesWithHighestActivation(
    layer_data: dict,
    neuron_index: int,
    top_k: int = 5,
    plot: bool = False,
) -> list[torch.Tensor]:
    
    if "images" not in layer_data or "sae_z" not in layer_data:
        raise KeyError("layer_data must contain 'images' and 'sae_z' keys.")

    images = layer_data["images"]   # (N, 3, H, W)
    sae_z  = layer_data["sae_z"]    # (N, F)

    if neuron_index >= sae_z.size(1):
        raise IndexError(f"neuron_index {neuron_index} out of bounds for F={sae_z.size(1)}")

    scores = sae_z[:, neuron_index]
    k = min(top_k, scores.size(0))
    top_idx = torch.topk(scores, k=k, largest=True).indices

    if not plot:
        return [images[i] for i in top_idx]
    else:
        fig, axes = plt.subplots(1, k, figsize=(k * 3, 3))
        if k == 1:
            axes = [axes]
        for ax, idx in zip(axes, top_idx):
            img = images[idx].permute(1, 2, 0).cpu().numpy()  # CHW → HWC
            ax.imshow(img)
            ax.axis("off")
        fig.suptitle(f"Top {k} images – latent #{neuron_index}")
        plt.tight_layout()
        plt.show()   
        
def plotActivation(activations):
    
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu()
        if activations.ndim == 2 and (activations.shape == (1, 768) or activations.shape == (768, 1)):
            activations = activations.flatten()
        else:
            activations = activations.numpy()
    elif isinstance(activations, np.ndarray):
        if activations.ndim == 2 and (activations.shape == (1, 768) or activations.shape == (768, 1)):
            activations = activations.flatten()

    plt.style.use(['science', 'no-latex'])

    x = np.arange(len(activations))
    y = activations

    color_values = np.abs(y)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = mcolors.Normalize(color_values.min(), color_values.max())
    lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(color_values)
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)

    ax.set_xlabel('Latent Dimension', fontsize=12)
    ax.set_ylabel('Activation (z)', fontsize=12)
    ax.set_title('Activations across Dimensions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()