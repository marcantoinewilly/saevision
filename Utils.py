import torch
import clip
from tqdm import tqdm
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
            transforms.Resize((resizew, resizeh)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), 0


def createImageDataset(path, resizew, resizeh):
    return FlatImageDataset(path, resizew, resizeh)


def createImageDataloader(path, resizew, resizeh, bsize=64, shuffle=True):
    dataset = createImageDataset(path, resizew, resizeh)
    dataloader = DataLoader(dataset, batch_size=bsize, shuffle=shuffle, num_workers=2)
    return dataloader


def getAverageViTActivation(train_loader, device, samples=16_000, clip_model_name="ViT-B/16"):
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
    clip_model.eval()

    activations = []
    total_collected = 0
    hooked_cls_output = {}

    target_block = clip_model.visual.transformer.resblocks[-2]

    def hook(module, input, output):
        hooked_cls_output["cls"] = output[:, 0, :]

    handle = target_block.register_forward_hook(hook)

    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc="Collecting ViT activations"):
            images = images.to(device)
            images_preprocessed = torch.stack([clip_preprocess(img) for img in images]).to(device)

            hooked_cls_output.clear()
            _ = clip_model.visual(images_preprocessed)

            if "cls" not in hooked_cls_output:
                raise RuntimeError("CLS token not captured from -2 layer.")

            cls_tokens = hooked_cls_output["cls"].float()
            activations.append(cls_tokens)
            total_collected += cls_tokens.size(0)

            if total_collected >= samples:
                break

    handle.remove()

    all_activations = torch.cat(activations, dim=0)[:samples]
    mean_activation = all_activations.mean(dim=0)

    return mean_activation