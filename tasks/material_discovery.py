# This script generates the plots for material discovery, specifically the plots in Fig. 3bc

import pickle
import argparse
import sys

sys.path.append('.')  
sys.path.append('../')      

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.model.PotNet.models.potnet import PotNet
from src.model.transformer_dos import TransformerDOS
from src.data.materials_project.dataset.dataset import MatDataset
from src.data.materials_project.dataset.collate_functions import collate
from src.utils.utils import fix_seed, switch_mode
from config.potnet_config import potnet_config

parser = argparse.ArgumentParser(description='Material discovery')

# general
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--checkpoint_to_load', type=str, default='model.pt', help='Path to checkpoint to load')
parser.add_argument('--projectors', action=argparse.BooleanOptionalAction, default=False, help='If checkpoint has projectors')      # NOTE: It's important to get this argument right and can be easy to miss 
parser.add_argument('--batch_size', type=int, default=32) 
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--num_neighbours', type=int, default=25, help='Number of nearest neighbours to find')

# model general
parser.add_argument('--latent_dim', type=int, default=128)

# DOS encoder 
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--dim_head', type=int, default=64)


class MatDiscoveryDataset(torch.utils.data.Dataset):
        def __init__(
                self,
                material_ids, 
                material_discovery_modality_data_path):
                    
            # load data
            self.data = torch.load(material_discovery_modality_data_path)

            # used if we only want to use a subset of the materials
            if material_ids is not None:
                self.data = {material_id: self.data[material_id] for material_id in material_ids}

            self.material_ids = list(self.data.keys())
    
        def __len__(self):
            return len(self.material_ids)
    
        def __getitem__(self, idx):
            material_id = self.material_ids[idx]
            return self.data[material_id]
        

def eval_material_discovery(
        material_ids_targets,
        materials_ids_nearest_neighbours,
        range_interpolation=5, 
        n=1, 
        best=True,
        dos_path=f'./data/dos.pt'):
    """
    material_ids_target are the material ids of the material discovery modality (dos)
    materials_ids_nearest_neighbours are the material ids of the nearest neighbour crystals for each dos sample
    n is the number of nearest neighbors to consider. 
    If best is True, then we select the best of n nearest neighbors and otherwise the nth nearest neighbor."""

    # load DOS
    dos_data = torch.load(dos_path)

    # find number of points typically in the range for which we calculate normalized MAE
    # this is used as the number of points in the range when interpolation is done
    total_num_points = 0
    for mpid in dos_data.keys():
        mask = (dos_data[mpid][0, :] >= -range_interpolation) & (dos_data[mpid][0, :] <= range_interpolation)
        num_points = torch.sum(mask)
        total_num_points += num_points
    avg_num_points = total_num_points / len(dos_data.keys())
    energies = torch.linspace(-range_interpolation, range_interpolation, int(avg_num_points))

    # compute normalized MAR between target DOS and the DOS corresponding to the nearest neighbors crystals
    # MAE is normalized by area of target 
    n_mae_values = []
    for i, material_id in enumerate(material_ids_targets):
        dos_target = dos_data[material_id]
        dos_target_interp = torch.from_numpy(np.interp(energies, dos_target[0, :], dos_target[1, :]))
        area_target = torch.trapz(dos_target_interp, energies)  # TODO correct

        if best:
            material_ids_nn = materials_ids_nearest_neighbours[i][:n]

            # compute MSE between target and each of the n nearest neighbors and use the best one
            best_n_mae = float('inf')
            for mpid_nn in material_ids_nn:
                dos_nn = dos_data[mpid_nn]
                dos_nn_interp = torch.from_numpy(np.interp(energies, dos_nn[0, :], dos_nn[1, :]))

                mae = torch.trapz(torch.abs(dos_target_interp - dos_nn_interp), energies)
                n_mae = mae / area_target

                if n_mae < best_n_mae:
                    best_n_mae = n_mae

            n_mae_append = best_n_mae
        else:
            dos_nn = dos_data[materials_ids_nearest_neighbours[i][n-1]]
            dos_nn_interp = torch.from_numpy(np.interp(energies, dos_nn[0, :], dos_nn[1, :]))

            mae = torch.trapz(torch.abs(dos_target_interp - dos_nn_interp), energies)
            n_mae = mae / area_target
            n_mae_append = n_mae

        # For ~7 DOS samples, the area of the target DOS is zero which causes the normalized MSE 
        # to be NaN or Inf. These samples are excluded
        if area_target == 0:
            continue

        n_mae_values.append(n_mae_append)

    avg_n_mae = torch.mean(torch.tensor(n_mae_values))
    return avg_n_mae

    
def main():

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modalities = ['crystal', 'dos']

    # dataset
    fix_seed(args.seed)
    modalities_to_include = ['crystal', 'dos', 'charge_density', 'bandgap', 'efermi', 'eform']
    dataset = MatDataset(
        modalities=modalities_to_include,
        mask_non_intersect=False) 

    # split dataset
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [0.8, 0.0, 0.2])   

    # get material ids for test set and train set
    crystal_ids_test = [dataset.keys[i] for i in test_dataset.indices]
    crystal_ids_train = [dataset.keys[i] for i in train_dataset.indices]

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=collate
    )

    crystal_encoder = PotNet(potnet_config)
    dos_encoder = TransformerDOS(dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head, mlp_dim=4 * args.dim)

    # dictionary with encoders
    encoders = {'crystal': crystal_encoder, 'dos': dos_encoder}

    # projectors
    projectors = {
        'crystal': nn.Linear(args.latent_dim, args.latent_dim, bias=False), 
        'dos': nn.Linear(args.latent_dim, args.latent_dim, bias=False)}

    # load CLIP checkpoint
    saved_state_dict = torch.load(args.checkpoint_to_load, map_location=torch.device('cpu'))
    for modality in modalities:
        encoders[modality].load_state_dict(saved_state_dict[f'{modality}_state_dict'])

        if args.projectors:
            projectors[modality].load_state_dict(saved_state_dict[f'projection_matrix_' + modality + '_state_dict'])
            encoder_with_projector = nn.Sequential(encoders[modality], projectors[modality])
            encoders[modality] = encoder_with_projector
        
        encoders[modality] = encoders[modality].to(device)

    print(f'loaded pretrained encoders from path {args.checkpoint_to_load}')
 
    # get representations (embeddings) of crystals
    representations = []

    # declare eval mode
    switch_mode(modalities, encoders, None, mode='eval')

    # get representations
    with torch.no_grad():
        for data in data_loader:  
            # get embeddings
            crystals = data['crystal'].to(device)
            z = encoders['crystal'](crystals)
            representations.append(z.to('cpu'))

    # convet to torch tensor and L2 normalize across samples (as done in the CLIP loss)
    representations = torch.cat(representations, dim=0)

    # get data loader for material discovery modality (dos)
    dos_path = f'./data/dos.pt'
    dataset_mat_discovery = MatDiscoveryDataset(material_ids=crystal_ids_test, material_discovery_modality_data_path=dos_path)
    
    data_loader_mat_discovery = torch.utils.data.DataLoader(
        dataset_mat_discovery,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # get representations (embeddings) for material discovery modality
    representations_mat_discovery = []
    with torch.no_grad():
        for data_mat_discovery in data_loader_mat_discovery:
            data_mat_discovery = data_mat_discovery.to(device)

            # get embeddings
            z_mat_discovery_modality = encoders['dos'](data_mat_discovery)
            representations_mat_discovery.append(z_mat_discovery_modality.to('cpu'))
    
    # convert to torch tensor
    representations_mat_discovery = torch.cat(representations_mat_discovery, dim=0)

    assert representations_mat_discovery.dim() == 2

    # nearest neighbor search (for all different samples in the test set from the material discovery modality)
    # all crystals from the train set are considered as neighbours/candidates
    closest_crystal_all_sampels = []
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    for i in range(len(dataset_mat_discovery)):
        similarity = cosine_similarity(representations_mat_discovery[i:i+1, :], representations)

        _, indices = torch.topk(similarity, k=args.num_neighbours, largest=True, sorted=True)

        closest_crystal_ids = [crystal_ids_train[int(idx)] for idx in indices]
        closest_crystal_all_sampels.append(closest_crystal_ids)

    # save material ids for the material discovery modality (dos) (from the test set) and the the 
    # corresponding nearest neighbor crystals (from the trains set)
    with open('material_ids_targets_and_nn_material_discovery.pkl', 'wb') as f:
        pickle.dump((dataset_mat_discovery.material_ids, closest_crystal_all_sampels), f)

    material_ids_targets = dataset_mat_discovery.material_ids
    material_ids_nearest_neighbours = closest_crystal_all_sampels

    # compute normalized MAE between target DOS and DOS corresponding to candidate material (i.e. data in Fig 3b)
    avg_n_mae_vals = []
    max_num_neighbours = 15
    for n_neighbours in range(1, max_num_neighbours + 1):
        # we must search for more neighbours than we compute the best normalized MAE for
        assert n_neighbours < args.num_neighbours 

        avg_n_mae = eval_material_discovery(
            material_ids_targets, 
            material_ids_nearest_neighbours, 
            range_interpolation=5,
            n=n_neighbours, 
            best=True, 
            dos_path=f'./data/dos.pt')

        avg_n_mae_vals.append(avg_n_mae)

    with open('eval_material_discovery.pkl', 'wb') as f:
        pickle.dump((avg_n_mae_vals), f)

    # make Fig. 3b plot using the avg_n_mae_vals
    neighbors = np.arange(1, max_num_neighbours + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(neighbors, avg_n_mae_vals, color='red', linestyle='-', marker='s', markersize=4)

    ax.set_title(f'Normalized MAE between target DOS and DOS corresponding to candidate material \n when using the best candidate out of the N closest neighbours')
    ax.set_xlabel('Number of closest neighbours considered (N)')
    ax.set_ylabel('MAE')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'normalized_MAE_material_discovery.pdf', dpi=300)
    plt.show()

    # plot two examples of the target DOS and the DOS corresponding to the best candidate material (i.e. the closest neighbour)
    mpids_targets = ['mp-1197252', 'mp-768042']
    
    # NOTE: The nearest neighbours for the above targets (using our model) are the below two materials. 
    # This is what's displayed in teh figure of the paper
    # nearest_neighbours = ['mp-1199023', 'mp-767960']

    dos_data = torch.load(dos_path)

    for i, mpid_target in enumerate(mpids_targets):
        target_idx = material_ids_targets.index(mpid_target)
        nearest_neigbour_id = material_ids_nearest_neighbours[target_idx][0]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(dos_data[mpid_target][0,:], dos_data[mpid_target][1,:], label='Target DOS', color='blue', linewidth=2)
        ax.plot(dos_data[nearest_neigbour_id][0,:], dos_data[nearest_neigbour_id][1,:], label='DOS of best candidate material', color='red', linestyle='--', linewidth=2)
        
        ax.set_title('Target vs. nearest neighbour DOS')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Density of States')
        ax.legend()
        ax.legend(fontsize='small')

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout() 

        plt.savefig(f'material_discovery_examples_{i}.pdf', dpi=300) 
        plt.show()


if __name__ == '__main__':
    main()
    print('Done!')
    