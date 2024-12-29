# This scripts produces the plots for interpretability of crystal embedings (i.e. Fig. 4 in the paper)

import argparse
import sys

sys.path.append('.')
sys.path.append('../')    

import numpy as np
import torch
import umap
import matplotlib
import matplotlib.pyplot as plt

from src.model.PotNet.models.potnet import PotNet
from src.data.materials_project.dataset.dataset import MatDataset
from src.data.materials_project.dataset.collate_functions import collate
from src.utils.utils import fix_seed, tensros_to_device
from config.potnet_config import potnet_config

parser = argparse.ArgumentParser(description='Interpretability')

# general
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--checkpoint_to_load', type=str, default='checkpoint.pth')
parser.add_argument('--distributed_checkpoint', action=argparse.BooleanOptionalAction, default=False, help='If checkpoint from distributed training')   # NOTE: It's important to get this argument right and can be easy to miss
parser.add_argument('--projectors', action=argparse.BooleanOptionalAction, default=False)   # NOTE: It's important to get this argument right and can be easy to miss
parser.add_argument('--latent_dim', type=int, default=128)

# data
parser.add_argument('--train_fraction', type=float, default=0.8)
parser.add_argument('--validation_fraction', type=float, default=0.0)
parser.add_argument('--test_fraction', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=64) 
parser.add_argument('--num_workers', type=int, default=16)

    
def main():

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    fix_seed(args.seed)
    dataset = MatDataset(
        modalities=['crystal', 'dos', 'charge_density', 'bandgap', 'efermi', 'eform'],
        mask_non_intersect=False) 

    # split dataset
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [args.train_fraction, args.validation_fraction, args.test_fraction])
    
    # get material ids for test set
    indices_test = test_dataset.indices
    material_ids_test = [dataset.keys[i] for i in indices_test]

    # dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=collate
    )

    encoder = PotNet(potnet_config)

    projector = torch.nn.Linear(args.latent_dim, args.latent_dim, bias=False)

    # load checkpoint
    saved_state_dict = torch.load(args.checkpoint_to_load, map_location=torch.device('cpu'))
    if args.distributed_checkpoint:
        saved_state_dict[f'crystal_state_dict'] = {k.replace('module.', ''): v for k, v in saved_state_dict[f'crystal_state_dict'].items()}

    encoder.load_state_dict(saved_state_dict[f'crystal_state_dict'])
    if args.projectors:
        projector.load_state_dict(saved_state_dict[f'projection_matrix_crystal_state_dict'])
        encoder = torch.nn.Sequential(encoder, projector)     

    encoder = encoder.to(device)
    print(f'loaded pretrained crystal encoder from path {args.checkpoint_to_load}')
 
    # store embeddings
    embeddings = []

    # declare eval mode
    encoder.eval()

    # get embeddings
    with torch.no_grad():
        for data in test_loader:
            # move tensors to device
            data = tensros_to_device(['crystal'], data, device)
            modality_embeddings = encoder(data['crystal'])
            embeddings.append(modality_embeddings.to('cpu'))

    # concatenate embeddings to form 2D tensors of shape (num_samples, latent_dim)
    embeddings = torch.cat(embeddings, dim=0)

    # dimensionality reduction (UMAP)
    reducer = umap.UMAP(
        n_neighbors=15,     # default: 15, small numbers emphasize local structure, larger global structure
        min_dist=0.1,       # default: 0.1, Minimum distance between embedded points. Smaller values lead to more clustered embeddings.
        n_components=2,     # Number of dimensions in which we want to embed the data. 2D is common for visualization.
        metric='cosine'     # The metric to use to compute distances in the original space. Can be other options like 'manhattan'
    )
    embeddings_umap = reducer.fit_transform(embeddings)

    # save embeddings
    torch.save(embeddings_umap, 'embeddings_umap.pt')
    torch.save(material_ids_test, 'material_ids_umap.pt')

    # data to color embeddings by 
    bandgap = torch.load('./data/bandgap.pt')
    eform = torch.load('./data/eform.pt')
    values_colour_bandgap = np.array([bandgap[mpid] for mpid in material_ids_test])
    values_colour_eform = [eform[mpid] for mpid in material_ids_test]
    threshold = 1e-9
    values_colour_metal = (values_colour_bandgap < threshold).astype(int)
    
    symmetry_dict = torch.load('./data/symmetry.pt')
    crystal_systems_colouring = [symmetry_dict[mpid]['Crystal System'] for mpid in material_ids_test]
    
    # matplotlib.use('Agg')
    # colour-code plot by formation energy ------------------------------------------------
    values_colour_eform = np.array(values_colour_eform)

    # A few materials are outliers. There are removed for a better visualization and clearer gradient.
    high = 2
    low = -3
    mask_high = (values_colour_eform > high)
    mask_low = (values_colour_eform < low)

    mask_remove = mask_high | mask_low
    values_colour_eform = values_colour_eform[~mask_remove]
    embeddings_umap_eform = embeddings_umap[~mask_remove, :]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(embeddings_umap_eform[:, 0], embeddings_umap_eform[:, 1], c=values_colour_eform, s=10, cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02)
    cbar.set_label('Formation energy', rotation=270, labelpad=10)

    plt.title('Dimensionality reduction of crystal embeddings colour-coded the formation energy')

    plt.tight_layout()
    plt.savefig('crystal_embeddings_umap_eform.pdf')
    plt.show()

    # colour-code plot by is metal ------------------------------------------------
    values_colour_metal = np.array(values_colour_metal)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=values_colour_metal, s=10, cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(scatter, orientation='vertical', pad=0.02)
    cbar.set_label('Is Metal', rotation=270, labelpad=10)

    plt.title('Dimensionality reduction of crystal embeddings colour-coded by whether the material is a metal')

    plt.tight_layout()
    plt.savefig('crystal_embeddings_umap_metal.pdf')
    plt.show()

    # colour-code plot by crystal system ------------------------------------------------

    # convert crystal systems to colours
    unique_crystal_systems = np.unique(crystal_systems_colouring)
    crystal_system_to_int = {crystal: i for i, crystal in enumerate(unique_crystal_systems)}
    crystal_system_indices = np.array([crystal_system_to_int[crystal] for crystal in crystal_systems_colouring])

    # Define a list of 7 distinct colors
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']

    # colour-code plot by crystal system
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=crystal_system_indices, s=10, cmap=matplotlib.colors.ListedColormap(colors), alpha=0.2)

    # Add a legend for crystal systems
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=unique_crystal_systems[i], 
                markerfacecolor=colors[i], markersize=10) for i in range(len(unique_crystal_systems))]
    plt.legend(handles=handles, title="Crystal Systems")

    plt.title('Dimensionality reduction of crystal embeddings colour-coded by crystal system')

    plt.tight_layout()
    plt.savefig('crystal_embeddings_umap_crystal_system.pdf')
    plt.show()


if __name__ == '__main__':
    main()
    print('Done!')
    