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
    def findBbias(self, loader, n_sample: int = 100_000):

        device = self.b.device
        xs, count = [], 0
        for batch in loader:
            x = batch[0] if isinstance(batch, (list,tuple)) else batch
            x = x.to(device)
            xs.append(x)
            count += x.shape[0]
            if count >= n_sample:
                break

        X = torch.cat(xs)[:n_sample]

        eps, max_it = 1e-6, 1000
        y = X[0].clone()
        for _ in range(max_it):
            dists = (X - y).norm(dim=1).clamp_min(eps)
            weights = 1.0 / dists
            y_next = (weights[:, None] * X).sum(0) / weights.sum()
            if (y_next - y).norm() < eps:
                break
            y = y_next

        self.b.data.copy_(y)
        print(f"[LOG] Bias initialised to Geometric Median of {len(X):,} Tokens")
        