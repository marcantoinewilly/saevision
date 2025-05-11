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

# https://arxiv.org/pdf/2407.14435 "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders"
class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, theta, h):
 
        ctx.save_for_backward(input, theta)
        ctx.h = h
        return torch.where(input >= theta, input, torch.zeros_like(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        h = ctx.h
        grad_input = grad_output * (input >= theta).type_as(grad_output)
        indicator = ((input >= theta - h/2) & (input <= theta + h/2)) \
                        .type_as(grad_output) / h
        grad_theta = (-grad_output * indicator).sum(dim=0)
        return grad_input, grad_theta, None

class HeavisideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, theta, h):
   
        ctx.save_for_backward(input, theta)
        ctx.h = h
        return (input >= theta).type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        h = ctx.h
        grad_input = None
        indicator = ((input >= theta - h/2) & (input <= theta + h/2)) \
                        .type_as(grad_output) / h
        grad_theta = (-grad_output * indicator).sum(dim=0)
        return grad_input, grad_theta, None, 

class JumpReLuSAE(nn.Module):
    def __init__(self, d, n, lam=1.0, h=0.05):

        super().__init__()
        self.d, self.n = d, n
        self.lam = lam
        self.h = h

        W_dec = torch.randn(d, n) / math.sqrt(d)
        W_dec = F.normalize(W_dec, dim=0)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())

        self.phi = nn.Parameter(torch.zeros(n))

    def forward(self, x):

        theta = F.softplus(self.phi)         
        pre_act = x @ self.W_enc.T           
        z = JumpReLUFunction.apply(pre_act, theta, self.h)
        recon = z @ self.W_dec.T             
        return recon, z, pre_act, theta

    def loss(self, x):
  
        recon, _, pre_act, theta = self.forward(x)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        mask = HeavisideFunction.apply(pre_act, theta, self.h)
        l0_loss = mask.sum()
        return recon_loss + self.lam * l0_loss

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def findTheta(self, vit, dataloader, n, device=None):

        vit.eval()
        device = device or next(self.parameters()).device

        all_pre = []
        for i, batch in enumerate(dataloader):
            if i >= n:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            feats = vit.encode_image(images)  
            pre_act = feats @ self.W_enc.T     
            all_pre.append(pre_act.cpu())

        if not all_pre:
            raise ValueError("No Batches processed... (n={n})")

        all_pre = torch.cat(all_pre, dim=0)      
        median = all_pre.median(dim=0).values    
 
        eps = 1e-6
        phi_init = torch.log(torch.clamp(torch.exp(median) - 1, min=eps))
        self.phi.data.copy_(phi_init.to(self.phi.device))
        print("[LOG] Initialized θ to Median")
        

class ReLuSAE(nn.Module):
    
    def __init__(self, d, n, lam=1.0):
    
        super().__init__()
        self.d, self.n = d, n
        self.lambda_ = lam

        W_dec = torch.randn(d, n) / math.sqrt(d)
        W_dec = F.normalize(W_dec, dim=0)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())

        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        
        pre = (x - self.b) @ self.W_enc.T
        z   = F.relu(pre)
        recon = z @ self.W_dec.T + self.b
        return recon, z, pre

    def loss(self, x):
        
        recon, z, _ = self.forward(x)
        rec_loss = F.mse_loss(recon, x, reduction='sum')
        l1_loss  = z.abs().sum()
        return rec_loss + self.lambda_ * l1_loss

    @torch.no_grad()
    def renorm(self): self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def findBias(self, vit, dataloader, n, device=None):
 
        vit.eval()
        device = device or next(self.parameters()).device

        feats = []
        for i, batch in enumerate(dataloader):
            if i >= n:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            f = vit.encode_image(images) 
            feats.append(f.cpu())
        if not feats:
            raise ValueError(f"No batches processed (n={n})")

        features = torch.cat(feats, dim=0) 
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