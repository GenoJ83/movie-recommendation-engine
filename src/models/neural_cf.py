"""Neural Collaborative Filtering (NCF) PyTorch Architecture."""
import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users: int, num_items: int, latent_dim: int = 32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, latent_dim)
        self.item_embed = nn.Embedding(num_items, latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_indices, item_indices):
        u = self.user_embed(user_indices)
        i = self.item_embed(item_indices)
        x = torch.cat([u, i], dim=-1)
        out = self.mlp(x)
        return out.squeeze(-1)
