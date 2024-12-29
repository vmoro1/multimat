# PotNet configuration file

from argparse import Namespace


potnet_config = Namespace(**{
    "conv_layers": 3,
    "rbf_min": -4.0,
    "rbf_max": 4.0,
    "potentials": [-0.801, -0.074, 0.145],      # coefficients for infinite summations; should be negative, negative, positive w.r.t. their mathematical form
    "charge_map": False,        # if including information of periodic table
    "transformer": False,       # enable transformer structure for infinite potential summation; only works when euclidean is False
    "atom_input_features": 92,
    "euclidean": False,
    'inf_edge_features': 64,
    'fc_features': 256,         
    'embedding_dim': 128,
    "projector": False,
    "final_bn": False
})