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

plt.rcParams.update({
    "figure.figsize": (10, 6),   # bigger plots for interactive inspection
    "figure.dpi":     110,       # crisper text in Jupyter/Colab
})

# Lightweight Dataset that applies CLIP Preprocessing to every Image File
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

# Convenience Wrapper that returns a ClipImageDataset
def createImageDataset(
    path: str,
    model_name: str = "openai/clip-vit-base-patch32"
) -> ClipImageDataset:

    processor = CLIPProcessor.from_pretrained(model_name)
    return ClipImageDataset(path, processor)

# Builds a DataLoader for the preprocessed Image Dataset
def createImageDataloader(
    path: str,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:

    dataset = createImageDataset(path, model_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True
    )

# Counts SAE Latent Units that are never non‑zero on the Loader
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

# Caches Images, ViT Activations, SAE Latents & Reconstructions
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

# Returns / Plots the TopK Images that maximally activate a Latent
def findImagesWithHighestActivation(
    layer_data: dict,
    neuron_index: int,
    top_k: int = 5,
    plot: bool = False,
    denorm: bool = False,
    figsize: tuple | None = None,
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
        fig, axes = plt.subplots(1, k, figsize=figsize) if figsize else plt.subplots(1, k)
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
        fig.suptitle(f"Top {k} Images for Latent #{neuron_index}")
        # avoid large gap between suptitle and images
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()   
        
# Colour‑line Plot of a 1-D Activation Vector
def plotActivation(activations, figsize: tuple | None = None):
    
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

    fig, ax = plt.subplots(figsize=figsize) if figsize else plt.subplots()
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)

    ax.set_xlabel('Latent Dimension', fontsize=12)
    ax.set_ylabel('Activation (z)', fontsize=12)
    ax.set_title('Activations across Dimensions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

# Histogram of Activation Values for one Latent Neuron
def plotLatentHistogram(
    layer_data: dict,
    neuron_index: int,
    bins: int | None = None,
    log_y: bool = True,
    int_bins: bool = False,
    figsize: tuple | None = None,
    ):
    """
    Plots a histogram of activation values for one latent neuron.
    If bins is None, chooses bins automatically:
      - If int_bins is True and min/max values are integers, uses integer-aligned bins.
      - Otherwise falls back to 'auto'.
    """
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

    # Choose bins automatically when bins is None
    if bins is None:
        if int_bins:
            v_min, v_max = values.min(), values.max()
            if float(v_min).is_integer() and float(v_max).is_integer():
                bins = np.arange(v_min - 0.5, v_max + 1.5)
            else:
                bins = "auto"
        else:
            bins = "auto"
    elif int_bins:
        # bins given but user still wants integer alignment
        v_min, v_max = values.min(), values.max()
        if float(v_min).is_integer() and float(v_max).is_integer():
            bins = np.arange(v_min - 0.5, v_max + 1.5)

    plt.figure(figsize=figsize) if figsize else plt.figure()
    plt.hist(values, bins=bins, color="steelblue", rwidth=1.0)
    if log_y:
        plt.yscale("log")

    plt.xlabel("Activation Value")
    plt.ylabel("Count")
    plt.title(f"Activation Histogram for Latent #{neuron_index}")
    plt.tight_layout()
    plt.show()

# Histogram of Number of Active Latents per Image
def plotActiveFeatureHistogram(
    layer_data: dict,
    bins: int | None = None,
    log_y: bool = False,
    int_bins: bool = True,
    figsize: tuple | None = None,
    ):
    
    if "sae_z" not in layer_data:
        raise KeyError("layer_data requires the key 'sae_z'.")

    z = layer_data["sae_z"]                  # (N, F)
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu()

    active_per_img = (z != 0).sum(dim=1).numpy()

    if bins is None:
        if int_bins:
            min_n, max_n = active_per_img.min(), active_per_img.max()
            bins = np.arange(min_n - 0.5, max_n + 1.5)
        else:
            bins = "auto"
    elif int_bins:
        min_n, max_n = active_per_img.min(), active_per_img.max()
        bins = np.arange(min_n - 0.5, max_n + 1.5)

    plt.figure(figsize=figsize) if figsize else plt.figure()
    plt.hist(active_per_img, bins=bins, color="seagreen", rwidth=1.0)
    if log_y:
        plt.yscale("log")

    plt.xlabel("# Active Features")
    plt.ylabel("Images")
    plt.title("Active Features per Image")
    plt.tight_layout()
    plt.show()

# Correlation of one Latent with every other Latent
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

# Lists the most strongly correlated Latent Pairs for a Reference Neuron
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

# Shows the pixel‑wise Mean of the TopK activating Images for a Latent
@torch.no_grad()
def plotAverageFeatureImage(
    layer_data: dict,
    neuron_index: int,
    top_k: int = 50,
    plot: bool = True,
    denorm: bool = False,
    figsize: tuple | None = None,
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
        plt.figure(figsize=figsize) if figsize else plt.figure()
        if denorm:
            mean = torch.tensor([0.481, 0.457, 0.408]).view(3,1,1)
            std  = torch.tensor([0.269, 0.261, 0.276]).view(3,1,1)
            img_denorm = (mean_img * std + mean).clamp(0,1)
            img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = mean_img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(f"Average of Top{k} Images for Latent #{neuron_index}")
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

def saveSAE(sae: nn.Module, path: str):
    torch.save(sae.state_dict(), path)
    print(f"[LOG] state_dict saved → {path}")

def loadSAE(sae: nn.Module, path: str, map_location: str | torch.device = "cpu"):
    state = torch.load(path, map_location=map_location)
    sae.load_state_dict(state)
    print(f"[LOG] state_dict loaded ← {path}")

# Return Indices of active Latents in a Vector or Batch
@torch.no_grad()
def getActiveLatents(
    z: torch.Tensor,
    threshold: float = 0.0,
    return_values: bool = False,
):
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)

    if z.ndim == 1:
        mask = z.abs() > threshold
        idx  = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if return_values:
            vals = z[mask].tolist()
            return list(zip(idx, vals))
        return idx
    elif z.ndim == 2:
        out = []
        for row in z:
            mask = row.abs() > threshold
            out.append(torch.nonzero(mask, as_tuple=False).flatten().tolist())
        return out
    else:
        raise ValueError("z must be 1-D or 2-D Tensor")