import torch
import torch.nn as nn
import torch.nn.functional as F

class TextMLP(nn.Module):
    ''' Projector network accepts a variable number of layers indicated by depth.
    Option to include batchnorm after every layer.'''

    def __init__(self, input_dim=768, latent_dim=128, non_linear =False, hidden_dim=512, bnorm=False, depth = 3):
        super(TextMLP, self).__init__()
        if non_linear:
            nlayer = [nn.BatchNorm1d(hidden_dim)] if bnorm else []
            list_layers = [nn.Linear(input_dim, hidden_dim)] + nlayer + [nn.ReLU()]
            for _ in range(depth-2):
                list_layers += [nn.Linear(hidden_dim, hidden_dim)] + nlayer + [nn.ReLU()]
            list_layers += [nn.Linear(hidden_dim, latent_dim)]
            self.mlp_block = nn.Sequential(*list_layers)
        else:
            self.mlp_block = nn.Linear(input_dim, latent_dim)
            

    def forward(self, x):
        x = self.mlp_block(x)
        return x
    