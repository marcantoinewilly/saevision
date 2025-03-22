import torch
from tqdm import tqdm
import clip

def trainSAEonViT(model, optimizer, train_loader, clip_model, clip_preprocess, device, alpha=1e-4, epochs=50):

    clip_model.eval()

    hooked_cls_output = {}

    target_block = clip_model.visual.transformer.resblocks[-2]

    def hook(module, input, output):
        hooked_cls_output["cls"] = output[:, 0, :]

    handle = target_block.register_forward_hook(hook)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = images.to(device)

            with torch.no_grad():
                images_preprocessed = torch.stack([clip_preprocess(img) for img in images]).to(device)

                hooked_cls_output.clear()

                _ = clip_model.visual(images_preprocessed)

                if "cls" not in hooked_cls_output:
                    raise RuntimeError("CLS token not captured. Ensure the hook is working correctly.")

                cls_tokens = hooked_cls_output["cls"].float()

            optimizer.zero_grad()
            x, z, xprime = model(cls_tokens)
            loss = model.loss(x, xprime, z, alpha)
            
            loss.backward()
            model.normalizeWdec()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tBatch Loss: {loss.item():.6f}')

        avg_loss = epoch_loss / len(train_loader)
        print(f'>> Epoch {epoch} complete. Avg Loss: {avg_loss:.6f}\n')

    handle.remove()