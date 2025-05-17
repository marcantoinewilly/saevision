import torch
import torch.nn as nn
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import scienceplots

# Lightweight dataset that applies CLIP preprocessing to every image file
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


# Convenience wrapper that returns a ClipImageDataset
def createImageDataset(
    path: str,
    model_name: str = "openai/clip-vit-base-patch32"
) -> ClipImageDataset:

    processor = CLIPProcessor.from_pretrained(model_name)
    return ClipImageDataset(path, processor)


# Builds a DataLoader for the preprocessed image dataset
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

# Counts SAE latent units that are never non‑zero on the loader
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

# Caches images, ViT activations, SAE latents & reconstructions
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

# Returns / plots the top‑k images that maximally activate a latent
def findImagesWithHighestActivation(
    layer_data: dict,
    neuron_index: int,
    top_k: int = 5,
    plot: bool = False,
    denorm: bool = False,
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
            img = images[idx]
            if denorm:
                mean = torch.tensor([0.481, 0.457, 0.408]).view(3,1,1)
                std  = torch.tensor([0.269, 0.261, 0.276]).view(3,1,1)
                img_denorm = (img * std + mean).clamp(0,1)
                img_np = img_denorm.permute(1,2,0).cpu().numpy()
            else:
                img_np = img.permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_np)
            ax.axis("off")
        fig.suptitle(f"Top {k} images – latent #{neuron_index}")
        plt.tight_layout()
        plt.show()   
        
# Colour‑line plot of a 1‑D activation vector
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

# Histogram of activation values for one latent neuron
def plotLatentHistogram(
    layer_data: dict,
    neuron_index: int,
    bins: int = 60,
    log_y: bool = True,
    ):
    
    if "sae_z" not in layer_data:
        raise KeyError("layer_data requires the key 'sae_z'.")

    z = layer_data["sae_z"]             # (N, F)
    if neuron_index >= z.size(1):
        raise IndexError(
            f"neuron_index {neuron_index} out of range (F={z.size(1)})"
        )

    values = z[:, neuron_index]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    plt.figure(figsize=(4, 3))
    plt.hist(values, bins=bins, color="steelblue")
    if log_y:
        plt.yscale("log")

    plt.xlabel("Activation Value")
    plt.ylabel("Count")
    plt.title(f"Histogram – Latent #{neuron_index}")
    plt.tight_layout()
    plt.show()


# Histogram of number of active latents per image
def plotActiveFeatureHistogram(
    layer_data: dict,
    bins: int = 40,
    log_y: bool = False,
    ):
    if "sae_z" not in layer_data:
        raise KeyError("layer_data requires the key 'sae_z'.")

    z = layer_data["sae_z"]                  # (N, F)
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu()

    active_per_img = (z != 0).sum(dim=1).numpy()

    plt.figure(figsize=(4, 3))
    plt.hist(active_per_img, bins=bins, color="seagreen")
    if log_y:
        plt.yscale("log")

    plt.xlabel("# Active Features")
    plt.ylabel("Images")
    plt.title("Active Features per Image")
    plt.tight_layout()
    plt.show()

# Correlation of one latent with every other latent (optionally plotted)
@torch.no_grad()
def plotNeuronCorrelation(
    layer_data: dict,
    neuron_index: int,
    plot: bool = True,
):
    
    if "sae_z" not in layer_data:
        raise KeyError("layer_data requires key 'sae_z'.")

    z = layer_data["sae_z"].float()         # (N, F)
    if neuron_index >= z.size(1):
        raise IndexError(
            f"neuron_index {neuron_index} out of range (F={z.size(1)})"
        )

    z_centered = z - z.mean(0)
    std = z.std(0) + 1e-8
    z_norm = z_centered / std

    ref = z_norm[:, neuron_index]           # (N,)
    corr_vec = (ref.unsqueeze(1) * z_norm).mean(0)  # (F,)

    corr_np = corr_vec.cpu().numpy()

    if not plot:
        return corr_np

    plotActivation(corr_np)

# Lists the most strongly correlated latent pairs for a reference neuron
@torch.no_grad()
def getCorrelatedNeurons(
    layer_data: dict,
    neuron_index: int,
    top_k: int = 10,
    min_abs_r: float = 0.2,
    ) -> list[tuple[int, float]]:
   
    if "sae_z" not in layer_data:
        raise KeyError("layer_data requires key 'sae_z'.")

    z = layer_data["sae_z"].float()         # (N, F)
    N, F = z.shape
    if neuron_index >= F:
        raise IndexError(
            f"neuron_index {neuron_index} out of range (F={F})"
        )

    z_std = (z - z.mean(0)) / (z.std(0) + 1e-8)

    ref = z_std[:, neuron_index]            # (N,)
    corr = (z_std.T @ ref) / N              # (F,)
    corr[neuron_index] = 0.0                # ignore self‑corr

    mask = corr.abs() >= min_abs_r
    idx  = torch.nonzero(mask, as_tuple=False).flatten()
    vals = corr[idx]

    order = torch.argsort(vals.abs(), descending=True)
    idx   = idx[order][:top_k].cpu().tolist()
    vals  = vals[order][:top_k].cpu().tolist()

    return list(zip(idx, vals))

# Shows the pixel‑wise mean of the top‑k activating images for a latent
@torch.no_grad()
def plotAverageFeatureImage(
    layer_data: dict,
    neuron_index: int,
    top_k: int = 50,
    plot: bool = True,
    denorm: bool = False,
    ):
    
    if "images" not in layer_data or "sae_z" not in layer_data:
        raise KeyError("layer_data requires keys 'images' and 'sae_z'.")

    images = layer_data["images"]   # (N,3,H,W)
    sae_z  = layer_data["sae_z"]    # (N,F)

    if neuron_index >= sae_z.size(1):
        raise IndexError(
            f"neuron_index {neuron_index} out of range (F={sae_z.size(1)})"
        )

    scores  = sae_z[:, neuron_index]
    k       = min(top_k, scores.size(0))
    top_idx = torch.topk(scores, k=k, largest=True).indices

    mean_img = images[top_idx].float().mean(dim=0)   # (3,H,W)

    if plot:
        if denorm:
            mean = torch.tensor([0.481, 0.457, 0.408]).view(3,1,1)
            std  = torch.tensor([0.269, 0.261, 0.276]).view(3,1,1)
            img_denorm = (mean_img * std + mean).clamp(0,1)
            img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = mean_img.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(3, 3))
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(f"Average of Top‑{k} Images – Latent #{neuron_index}")
        plt.tight_layout()
        plt.show()
        return None
    else:
        return mean_img

def countParameters(model: nn.Module, only_trainable: bool = True) -> int:

    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def saveSAE(
    sae: nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict | None = None, 
    ):

    ckpt = {
        "sae_state": sae.state_dict(),
    }
    if optimizer is not None:
        ckpt["optim_state"] = optimizer.state_dict()
    if extra is not None:
        ckpt["extra"] = extra

    torch.save(ckpt, path)
    print(f"[LOG] SAE Checkpoint written to {path}")

def loadSAE(
    sae: nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = "cpu",
    ):

    ckpt = torch.load(path, map_location=map_location)
    sae.load_state_dict(ckpt["sae_state"])

    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])

    print(f"[LOG] SAE Weights restored from {path}")
    return ckpt.get("extra", None)