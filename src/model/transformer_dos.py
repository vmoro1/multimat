# Transformer for density of states (DOS). The transformer used no positional encoding but used embeddings of the energy which are 
# then concatenated and mixed (by a linear layer) with the DOS embeddings.

import torch
from torch import nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # I can't see that torch.nn.functional.scaled_dot_product_attention exists but I'm using torch for mac and not the cuda version
        # out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerDOS(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., use_final_bn=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = nn.ModuleList([])
        self.energy_embedding = nn.Linear(1, dim)
        self.dos_embedding = nn.Linear(1, dim)
        self.reduce_embedding_dim = nn.Linear(2*dim, dim)
        if use_final_bn:
            self.bn = nn.BatchNorm1d(dim)
        else:
            self.bn = nn.Identity()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        energies = x[:,0,:].unsqueeze(-1)
        dos = x[:,1,:].unsqueeze(-1)

        tokens_energies = self.energy_embedding(energies)
        tokens_dos = self.dos_embedding(dos)
        tokens = torch.concatenate((tokens_energies, tokens_dos), dim=-1)
        tokens = self.reduce_embedding_dim(tokens)
        x = tokens

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = x.mean(dim=1)  # average pooling
        x = self.bn(x)
        return x

if __name__ == '__main__':
    x = torch.rand(2,2,512)
    model = TransformerDOS(dim=128, depth=2, heads=8, dim_head=64, mlp_dim=512, dropout=0)
    out = model(x)
    print(out.shape)
