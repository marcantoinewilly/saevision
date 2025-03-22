import torch
import clip
from tqdm import tqdm

def getAverageViTActivation(train_loader, device, samples = 16_000, clip="ViT-B/16"):
  
    clip_model, clip_preprocess = clip.load(clip, device=device)
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