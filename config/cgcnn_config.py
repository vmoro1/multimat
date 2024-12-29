# CGCNN configuration file

from argparse import Namespace


cgcnn_config = Namespace(**{
    'atom_fea_len': 64, # Should probably also change this when chaning embedding dim
    'n_conv': 3,
    'h_fea_len': 128,   # This is embedding dim
    'n_h': 1
})