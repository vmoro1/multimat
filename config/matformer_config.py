# config file for the matformer model. This is equivalent to the MatformerConfig class in the original codebase 
# (https://github.com/YKQ98/Matformer/blob/main/matformer/models/pyg_att.py#L17)

from argparse import Namespace

matformer_config = Namespace(**{
    'angle_lattice': False,
    'atom_input_features': 92,
    'classification': False,
    'conv_layers': 5,
    'edge_features': 128,
    'edge_layer_head': 4,
    'edge_layers': 0,
    'fc_features': 128,
    'fc_layers': 1,
    'link': 'identity',
    'name': 'matformer',
    'nn_based': False,
    'node_features': 128,
    'node_layer_head': 4,
    'output_features': 128,   # This is the latent dimension
    'triplet_input_features': 40,
    'use_angle': False,
    'zero_inflated': False
})