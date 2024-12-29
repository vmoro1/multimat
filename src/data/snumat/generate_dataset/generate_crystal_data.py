import sys
import json

from jarvis.core.atoms import Atoms
import torch

from src.model.matformer.graphs import PygGraph, PygStructureDataset

# loads the json file

file = open('./data/snumat.json')

# convert the data to a dictionary
data = json.load(file)


atom_features="cgcnn"

crystal = dict()
bandgap = dict()

for i,d  in enumerate(data):
    atoms = d['atoms']
    id = d['SNUMAT_id']
    bandgap[id] = d['Band_gap_HSE']

    crystal_strucuture_dict_to_jarvis = {
            'lattice_mat': atoms['lattice_mat'], 
            'coords': atoms['coords'], 
            'elements': atoms['elements'], 
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

    try: 
        z = graph.x
        graph.atomic_number = z
        z = z.type(torch.IntTensor).squeeze()
        f = torch.tensor(features[z]).type(torch.FloatTensor)
        if graph.x.size(0) == 1:
            f = f.unsqueeze(0)
        graph.x = f
        graph.edge_attr = graph.edge_attr.float()
        crystal[id] = graph
    except:
        print('Error with ', id)

    

    if i % 100 == 0: 
        print('Done with ', i, ' graphs')

# save the crystal dictionary
torch.save(crystal, './data/crystal_snumat.pt')
torch.save(bandgap, './data/bandgap_snumat.pt')
