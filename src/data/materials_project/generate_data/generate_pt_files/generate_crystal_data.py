# geenrate crystals

import sys

import h5py
from jarvis.core.atoms import Atoms
import torch

from src.model.matformer.graphs import PygGraph, PygStructureDataset


def _hdf5_to_dict(h5obj):
    if isinstance(h5obj, h5py.File):
        return _hdf5_to_dict(h5obj['/'])

    if isinstance(h5obj, h5py.Group):
        return {key: _hdf5_to_dict(h5obj[key]) for key in h5obj}

    if isinstance(h5obj, h5py.Dataset):
        return h5obj[()]

    raise ValueError("Unknown HDF5 object type: {}".format(type(h5obj)))


def generate_crystals():

    crystal_file = h5py.File('./mp_crystal.h5', 'r')
    material_ids = list(crystal_file.keys())
    atom_features="cgcnn"

    crystal_dict = {}   # Dictionary to store pre-processeed graphs
    for material_id in material_ids:
        crystal_strucuture = _hdf5_to_dict(crystal_file[material_id])
        coords_and_elements = [(atom[1]['abc'], atom[1]['label']) for atom in crystal_strucuture['sites'].items()]
        crystal_strucuture_dict_to_jarvis = {
            'lattice_mat': crystal_strucuture['lattice']['matrix'],
            'coords': [x[0] for x in coords_and_elements],
            'elements': [x[1].decode('utf-8') for x in coords_and_elements],
            'props': None,
            'cartesian': False,
            'show_props': False
        }

        structure = Atoms.from_dict(crystal_strucuture_dict_to_jarvis)
        graph = PygGraph.atom_dgl_multigraph(structure,
                                             neighbor_strategy='k-nearest',
                                             cutoff=8.0,
                                             atom_features="atomic_number",
                                             max_neighbors=12,
                                             compute_line_graph=False,
                                             use_canonize=True,
                                             use_lattice=True,
                                             use_angle=False)
        
        features = PygStructureDataset._get_attribute_lookup(atom_features)
        
        z = graph.x
        graph.atomic_number = z
        z = z.type(torch.IntTensor).squeeze()
        f = torch.tensor(features[z]).type(torch.FloatTensor)
        if graph.x.size(0) == 1:
            f = f.unsqueeze(0)
        graph.x = f
        graph.edge_attr = graph.edge_attr.float()  

        crystal_dict[material_id] = graph

    # save crystal graphs
    torch.save(crystal_dict, './crystal_new_correct.pt')


if __name__ == '__main__':
    generate_crystals()
    print('Done!')
