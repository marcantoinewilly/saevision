import torch.nn as nn
import torch
import math

class AE(nn.Module):
    # https://arxiv.org/pdf/2502.06755 (ReLu Autoencoder)

    def __init__(self, input_dim, hidden_dim):
        super(AE, self).__init__()
        
        # Encoder
        self.W_enc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        
        # Decoder
        self.W_dec = nn.Linear(hidden_dim, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        
        # Activation
        self.relu = nn.ReLU()

        # Loss
        self.mse = nn.MSELoss()
    
    def forward(self, x):

        z = self.relu(self.W_enc(x - self.b_dec) + self.b_enc)
        xprime = self.W_dec(z) + self.b_dec
        return x, z , xprime
    
    def loss(self, x, xprime):
        return self.mse(xprime, x)
    

class SAE(nn.Module):
    # https://arxiv.org/pdf/2502.06755 (Sparse ReLu Autoencoder)

    def __init__(self, input_dim, hidden_dim, b_dec_init):
        super(SAE, self).__init__()
        
        # Encoder
        self.W_enc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        
        # Decoder
        self.W_dec = nn.Linear(hidden_dim, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        
        # Initialization
        nn.init.kaiming_uniform_(self.W_enc.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec.weight, a=math.sqrt(5))
        self.b_enc.data.zero_()
        self.b_dec.data.copy_(b_dec_init)
        
        # Activation
        self.relu = nn.ReLU()

        # Loss
        self.mse = nn.MSELoss()

        # Sparsity Loss
        self.l1 = nn.L1Loss()
    
    def normalizeWdec(model):
        
        with torch.no_grad():
            W = model.W_dec.weight
            for i in range(W.shape[1]):
                col = W[:, i]
                norm = col.norm(p=2)
                if norm > 0:
                    W[:, i] = col / norm

        if model.W_dec.weight.grad is not None:
            G = model.W_dec.weight.grad
            W = model.W_dec.weight 
            for i in range(W.shape[1]):
                col = W[:, i]
                grad_col = G[:, i]
                parallel_component = torch.dot(grad_col, col) * col
                G[:, i] = grad_col - parallel_component
    
    def forward(self, x):

        z = self.relu(self.W_enc(x - self.b_dec) + self.b_enc)
        xprime = self.W_dec(z) + self.b_dec
        return x, z , xprime
    
    def loss(self, x, xprime, z, alpha):
        return self.mse(xprime, x) + alpha * self.l1(z, torch.zeros_like(z))
    

class OSAE(nn.Module):
    # Orthogonal Sparse ReLU Autoencoder (Work in Progress)
    
    def __init__(self, input_dim, hidden_dim, theta, rho, gamma):
        super(OSAE, self).__init__()

        # Encoder
        self.W_enc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))

        # Decoder
        self.W_dec = nn.Linear(hidden_dim, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        # Activation
        self.relu = nn.ReLU()

        # Loss
        self.mse = nn.MSELoss()
        
        # Hyperparameters
        self.theta = theta 
        self.rho = rho  
        self.gamma = gamma 

    def forward(self, x):

        z = self.relu(self.W_enc(x - self.b_dec) + self.b_enc)
        x_hat = self.W_dec(z) + self.b_dec 
        return x, z, x_hat

    def cosine_similarity(self, z):

        norm_z = torch.norm(z, p=2, dim=0, keepdim=True) + 1e-6 
        z_normalized = z / norm_z 
        C = torch.matmul(z_normalized.T, z_normalized)  
        C.fill_diagonal_(0) 
        return C

    def orthogonality_penalty(self, C):

        Wd = self.W_dec.weight
        C = C.clone()
        C.masked_fill_(C <= self.theta, 0)
        penalty = torch.norm((Wd.T @ Wd) * C, p='fro') ** 2 
        return penalty

    def loss(self, x, x_hat, z):

        mse_loss = self.mse(x_hat, x)
        sparsity_loss = self.rho * torch.sum(torch.abs(z)) 
        C = self.cosine_similarity(z)
        orthogonality_loss = self.gamma * self.orthogonality_penalty(C)
        return mse_loss + sparsity_loss + orthogonality_loss