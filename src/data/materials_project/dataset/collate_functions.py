# collate function to create a torch DataLoader. Most DataLoaders will use 'collate' below (e.g dataloaders for Matformer or 
# PotNet) but when using crystals not constructed with torch geometric we need differetn collate functions (e.g CGCNN)

import numpy as np
import torch
from torch_geometric.data.batch import Batch


def collate(samples):
    """Dataloader helper to batch graphs cross `samples`. `samples` is a list of dictionaries where each dictionary in on 
    the form as per the dataset. This one is used in most cases, e.g for Matformer or PotNet"""
    dict_to_return = {}
    for modality in samples[0].keys():
        data_moality = [d[modality] for d in samples]

        if modality == 'crystal':
            # batch across graphs
            batched_graphs = Batch.from_data_list(data_moality)
            dict_to_return[modality] = batched_graphs
        else:
            # data modality is a list of torch tensors or a list of floats. Depending on the case we need torch.stack or torch.tensor
            to_stack = True if type(data_moality[0]) == torch.Tensor else False

            if to_stack:
                dict_to_return[modality] = torch.stack(data_moality).float()
            else:
                dict_to_return[modality] = torch.tensor(data_moality).float()
    
    return dict_to_return

def collate_mask_non_int(samples):
    """Dataloader helper to batch graphs cross `samples`. `samples` is a list of tuples where each tuple contains two dictionaries: 'data' and 'exist_label'."""
    
    data_dict_to_return = {}
    exist_label_dict_to_return = {}

    for modality in samples[0][0].keys():
        data_modality = [d[0][modality] for d in samples]

        if modality == 'crystal':
            # batch across graphs
            batched_graphs = Batch.from_data_list(data_modality)
            data_dict_to_return[modality] = batched_graphs
        else:
            # data modality is a list of torch tensors or a list of floats. Depending on the case we need torch.stack or torch.tensor
            to_stack = True if type(data_modality[0]) == torch.Tensor else False

            if to_stack:
                data_dict_to_return[modality] = torch.stack(data_modality).float()
            else:
                data_dict_to_return[modality] = torch.tensor(data_modality).float()

    for modality in samples[0][1].keys():
        exist_label_modality = [d[1][modality] for d in samples]
        exist_label_dict_to_return[modality] = exist_label_modality

    return data_dict_to_return, exist_label_dict_to_return
  
def collate_cgcnn(samples):
    """
    Collate list of dictonaries containing data for CGCNN (i.e when using CGCNN architecture and crystals).

    Parameters
    ----------

    samples: list of dictionaries for each data material with keys being the modalities.
      For crystals, the values are tuples: (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    """
    dict_to_return = {}
    for modality in samples[0].keys():
        data_moality = [d[modality] for d in samples]

        if modality == 'crystal':
            # batch crystals
            batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
            crystal_atom_idx, batch_target = [], []
            batch_cif_ids = []
            base_idx = 0
            for (atom_fea, nbr_fea, nbr_fea_idx) in data_moality:
                n_i = atom_fea.shape[0]  # number of atoms for this crystal
                batch_atom_fea.append(atom_fea)
                batch_nbr_fea.append(nbr_fea)
                batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
                new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
                crystal_atom_idx.append(new_idx)
                base_idx += n_i

            dict_to_return['crystal'] = (torch.cat(batch_atom_fea, dim=0), 
                                         torch.cat(batch_nbr_fea, dim=0), 
                                         torch.cat(batch_nbr_fea_idx, dim=0), 
                                         crystal_atom_idx)
        else:
            # data modality is a list of torch tensors or a list of floats. Depending on the case we need torch.stack or torch.tensor
            to_stack = True if type(data_moality[0]) == torch.Tensor else False

            if to_stack:
                dict_to_return[modality] = torch.stack(data_moality).float()
            else:
                dict_to_return[modality] = torch.tensor(data_moality).float()
    
    return dict_to_return
  
def collate_mask_non_int_cgcnn(samples):
    """Dataloader helper to batch graphs cross `samples`. `samples` is a list of tuples where each tuple contains two dictionaries: 'data' and 'exist_label'."""
    
    data_dict_to_return = {}
    exist_label_dict_to_return = {}

    for modality in samples[0][0].keys():
        data_modality = [d[0][modality] for d in samples]

        if modality == 'crystal':
            # batch crystals
            batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
            crystal_atom_idx, batch_target = [], []
            batch_cif_ids = []
            base_idx = 0
            for (atom_fea, nbr_fea, nbr_fea_idx) in data_moality:
                n_i = atom_fea.shape[0]  # number of atoms for this crystal
                batch_atom_fea.append(atom_fea)
                batch_nbr_fea.append(nbr_fea)
                batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
                new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
                crystal_atom_idx.append(new_idx)
                base_idx += n_i

            data_dict_to_return['crystal'] = (torch.cat(batch_atom_fea, dim=0), 
                                         torch.cat(batch_nbr_fea, dim=0), 
                                         torch.cat(batch_nbr_fea_idx, dim=0), 
                                         crystal_atom_idx)
        else:
            # data modality is a list of torch tensors or a list of floats. Depending on the case we need torch.stack or torch.tensor
            to_stack = True if type(data_modality[0]) == torch.Tensor else False

            if to_stack:
                data_dict_to_return[modality] = torch.stack(data_modality).float()
            else:
                data_dict_to_return[modality] = torch.tensor(data_modality).float()

    for modality in samples[0][1].keys():
        exist_label_modality = [d[1][modality] for d in samples]
        exist_label_dict_to_return[modality] = exist_label_modality

    return data_dict_to_return, exist_label_dict_to_return