import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class TopKSAE(nn.Module):

    def __init__(self, d, n, k):
        super(TopKSAE, self).__init__()
        
        self.k = k
        self.d = d
        self.n = n
        
        # Initializing the Encoder as the Transpose of the Decoder helps to prevent dead Latents, according to OpenAI
        # https://arxiv.org/pdf/2406.04093v1
        W_dec = torch.randn(d, n) / math.sqrt(d)
        W_dec = F.normalize(W_dec, dim=0)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        
        print("[LOG] Bias initialized with 0-Vector")
        self.b  = nn.Parameter(torch.zeros(d))
    
    def topk(self, v):
        vals, idx = torch.topk(v, self.k, dim = -1)
        mask = torch.zeros_like(v).scatter_(-1, idx, 1.0)
        return v * mask

    def forward(self, x):
        z = (x - self.b) @ self.W_enc.T         # (B, n)
        z = self.topk(F.relu(z))                # (B, n)
        recon = z @ self.W_dec.T + self.b       # (B, d)
        return recon, z
    
    # Must be called after every Optimizer Step
    @torch.no_grad()
    def renorm(self): self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=0)
    
    # https://transformer-circuits.pub/2023/monosemantic-features
    # According to Antrophic, a learned Bias is a good Idea!
    @torch.no_grad()
    def findBias(self, vit, dataloader, n):
        
        vit.eval()
        device = next(vit.parameters()).device

        feats = []
        for i, batch in enumerate(dataloader):
            if i >= n:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            f = vit.encode_image(images)
            feats.append(f.cpu())
        if not feats:
            raise ValueError(f"No Batches processed... (n={n})")

        features = torch.cat(feats, dim=0) 
        N, d = features.shape

        median = features.mean(dim=0)

        eps = 1e-5
        max_iters = 500
        for _ in range(max_iters):
            diffs = features - median.unsqueeze(0)  
            dist = diffs.norm(dim=1)              

            zero_mask = dist < eps
            if zero_mask.any():
                median = features[zero_mask][0]
                break

            inv = 1.0 / (dist + eps)
            w = inv / inv.sum()                  
            new_med = (w.unsqueeze(1) * features).sum(dim=0)

            if (new_med - median).norm().item() < eps:
                median = new_med
                break
            median = new_med

        median = median.to(device)
        self.b.data.copy_(median)
        print("[LOG] Bias initialized to Geometric Median")