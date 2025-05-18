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
        return x, z, recon
    
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

class JumpReLUSAE(nn.Module):
    def __init__(self, d, n, lam=1.0, h=0.05):
        super().__init__()
        self.d, self.n = d, n
        self.lam = lam
        self.h   = h

        W_dec = torch.randn(d, n) / math.sqrt(d)
        W_dec = F.normalize(W_dec, dim=0)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())

        self.bias_d = nn.Parameter(torch.zeros(d))  
        self.bias_e = nn.Parameter(torch.zeros(n))  

        self.phi = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        theta = F.softplus(self.phi)           
        x_pre  = x - self.bias_d                
        pre_act = x_pre @ self.W_enc.T + self.bias_e
        z = JumpReLUFunction.apply(pre_act, theta, self.h)
        recon = z @ self.W_dec.T + self.bias_d  
        return x, z, recon, pre_act, theta

    def loss(self, x):
        recon, _, pre_act, theta = self.forward(x)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        mask       = HeavisideFunction.apply(pre_act, theta, self.h)
        l0_loss    = mask.sum()
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
            raise ValueError(f"No Batches processed... (n={n})")
        all_pre = torch.cat(all_pre, dim=0)
        median = all_pre.median(dim=0).values
        eps = 1e-6
        phi_init = torch.log(torch.clamp(torch.exp(median) - 1, min=eps))
        self.phi.data.copy_(phi_init.to(self.phi.device))
        print("[LOG] Initialized θ to Median")
        

class ReLUSAE(nn.Module):
    
    def __init__(self, input_dim: int, num_features: int, lam: float, alpha: float | None = None):
    
        super().__init__()
        self.input_dim, self.num_features = input_dim, num_features
        self.lambda_ = lam
        self.alpha = alpha  

        W_dec = torch.randn(input_dim, num_features) / math.sqrt(input_dim)
        W_dec = F.normalize(W_dec, dim=0)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())

        self.b = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        if self.alpha is not None:
            x = self.alpha * x

        pre   = (x - self.b) @ self.W_enc.T
        z     = F.relu(pre)
        recon = z @ self.W_dec.T + self.b
        return x, z, recon

    def loss(self, x):
        x_in, z, recon = self.forward(x)
        rec_loss = F.mse_loss(recon, x_in, reduction='sum')
        norms    = self.W_dec.norm(dim=0)
        l1_loss  = (z.abs() * norms).sum()
        return rec_loss + self.lambda_ * l1_loss

    @torch.no_grad()
    def renorm(self): self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def estimateAlpha(
        self,
        dataloader,
        feature_model,
        num_batches: int = 50,
        device=None,
        layer: int = -1,
    ) -> float:

        input_dim = self.input_dim

        feature_model.eval()
        device = device or next(feature_model.parameters()).device

        sq_sum, n_vecs = 0.0, 0

        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)

            if hasattr(feature_model, "encode_image"):
                try:
                    feats = feature_model.encode_image(images, cls_layer=layer)
                except TypeError:
                    feats = None
            else:
                feats = None

            if feats is None:
                out = feature_model(pixel_values=images, output_hidden_states=True)
                feats = out.hidden_states[layer][:, 0]  

            sq_sum += (feats ** 2).sum().item()
            n_vecs += feats.size(0)

        mean_sq = sq_sum / n_vecs
        self.alpha = math.sqrt(input_dim / mean_sq)
        return self.alpha

    @torch.no_grad()
    def findBias(
        self,
        vit,
        dataloader,
        n: int,
        layer: int = -1,
        device=None,
    ):
   
        vit.eval()
        device = device or next(self.parameters()).device

        feats = []
        for i, batch in enumerate(dataloader):
            if i >= n:
                break

            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)

            if hasattr(vit, "encode_image"):
                try:
                    f = vit.encode_image(images, cls_layer=layer)
                except TypeError:
                    f = vit.encode_image(images)
            else:
                out = vit(pixel_values=images, output_hidden_states=True)
                f = out.hidden_states[layer][:, 0]

            if self.alpha is not None:
                f = self.alpha * f

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

        self.b.data.copy_(median.to(device))
        print("[LOG] Bias initialized to Geometric Median")
        
class OrthogonalSAE(nn.Module):
    def __init__(self, input_dim: int, num_features: int,
                 sparsity: float = 0.04, orthogonality: float = 0.01,
                 theta: float = 0.7,
                 warmup_steps: int = 500,
                 ramp_steps: int = 1000,
                 theta_decay_steps: int = 1500):
        
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.sparsity = sparsity
        self.orthogonality = orthogonality
        self.theta = theta
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))
        # buffer for previous activations (temporal consistency)
        self.register_buffer('prev_f', torch.zeros(1))
        # weight of temporal consistency loss term
        self.temp_weight = 1e-3

        self.W_dec = nn.Parameter(torch.empty(input_dim, num_features))
        self.W_enc = nn.Parameter(torch.empty(num_features, input_dim))
        self.bias_e = nn.Parameter(torch.zeros(num_features))
        self.bias_d = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=0)
        self.W_enc.data[:] = self.W_dec.data.T.clone()

        self.warmup_steps      = warmup_steps
        self.ramp_steps        = ramp_steps
        self.theta_decay_steps = theta_decay_steps

    def forward(self, x: torch.Tensor):
 
        pre = x @ self.W_enc.T + self.bias_e   
        f = F.relu(pre)                        
        recon = f @ self.W_dec.T + self.bias_d
        return x, f, recon

    def compute_competition(self, f: torch.Tensor) -> torch.Tensor:
   
        mag = f.norm(dim=0, keepdim=True).clamp(min=1e-8)
        f_norm = f / mag                       
        C = f_norm.T @ f_norm                
        C = C - torch.diag(torch.diag(C))      
        return C

    def ortho_penalty(self, C: torch.Tensor) -> torch.Tensor:
  
        M = self.W_dec.T @ self.W_dec          
        M = M - torch.diag(torch.diag(M))    
        mask = (C > self.theta).type_as(M)       
        return torch.sum(mask * (M ** 2))      

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        
        warmup_steps      = self.warmup_steps
        ramp_steps        = self.ramp_steps
        theta_start       = 0.7
        theta_end         = 0.3
        theta_decay_steps = self.theta_decay_steps

        self.step += 1
        _, f, recon = self.forward(x)[:3]
        L_rec = F.mse_loss(recon, x, reduction='sum')

        # Phase 1 – only reconstruction
        if self.step <= warmup_steps:
            return L_rec

        # Phase 2 – ramp up sparsity
        ramp_end = warmup_steps + ramp_steps
        if self.step <= ramp_end:
            alpha = (self.step - warmup_steps) / ramp_steps
            return L_rec + alpha * self.sparsity * f.abs().sum()

        # Phase 3 – full loss: reconstruction + sparsity + orthogonality (+ temporal)
        L = L_rec + self.sparsity * f.abs().sum()

        # θ decay for competition mask
        decay_progress = min(1.0, (self.step - ramp_end) / theta_decay_steps)
        self.theta = theta_start - (theta_start - theta_end) * decay_progress

        # orthogonality term
        C = self.compute_competition(f)
        L += self.orthogonality * self.ortho_penalty(C)

        # temporal consistency term
        if self.prev_f.numel() != 1:
            L += self.temp_weight * F.mse_loss(f, self.prev_f, reduction='sum')
        self.prev_f = f.detach()

        return L

    @torch.no_grad()
    def renorm(self): self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=0)