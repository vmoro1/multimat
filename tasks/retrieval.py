# This script evaluate the retrieval performance of the model. It produces the plot in Fig. 3a of the paper.

import argparse
import itertools
import sys

sys.path.append('.')
sys.path.append('../')      

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.model.PotNet.models.potnet import PotNet
from src.model.transformer_dos import TransformerDOS
from src.model.ResNeXt_3D import resnext50
from src.data.materials_project.dataset.dataset import MatDataset
from src.data.materials_project.dataset.collate_functions import collate
from src.utils.utils import fix_seed, switch_mode, tensros_to_device
from config.potnet_config import potnet_config

parser = argparse.ArgumentParser(description='Retrieval')

# general
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--modalities', nargs='+', type=str, default=['crystal', 'dos', 'charge_density'], help='List of modalities for the CLIP checkpoint')
parser.add_argument('--checkpoint_to_load', type=str, default='checkpoint.pt', help='Path to checkpoint to load')
parser.add_argument('--projectors', action=argparse.BooleanOptionalAction, default=False, help='If checkpoint has projectors')  # NOTE: It's important to get this argument right and can be easy to miss 

# data
parser.add_argument('--train_fraction', type=float, default=0.8)
parser.add_argument('--validation_fraction', type=float, default=0.0)
parser.add_argument('--test_fraction', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=64) 
parser.add_argument('--num_workers', type=int, default=16)

# model general
parser.add_argument('--latent_dim', type=int, default=128)

# DOS encoder 
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--dim_head', type=int, default=64)

# Charge density encoder (3D ResNeXt)
parser.add_argument('--data_dim', type=int, default=32)
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--cardinality', type=int, default=32)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--shortcut_type', type=str, default='B')

    
def main():

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    fix_seed(args.seed)
    dataset = MatDataset(
        modalities=args.modalities+['bandgap', 'efermi', 'eform'],
        mask_non_intersect=False)

    # split dataset
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [args.train_fraction, args.validation_fraction, args.test_fraction])

    # dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=collate
    )

    crystal_encoder = PotNet(potnet_config)
    dos_encoder = TransformerDOS(dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head, mlp_dim=4 * args.dim)
    charge_density_encoder = resnext50(sample_depth=args.data_dim, 
                                sample_height=args.data_dim, 
                                sample_width=args.data_dim, 
                                shortcut_type=args.shortcut_type, 
                                cardinality=args.cardinality, 
                                in_channels=args.in_channels, 
                                embedding_dim=args.embedding_dim)

    # dictionary with encoders
    encoders_all = {'crystal': crystal_encoder, 'dos': dos_encoder, 'charge_density': charge_density_encoder}
    encoders = {modality: encoders_all[modality] for modality in args.modalities}

    # projectors
    projectors = {modality: nn.Linear(args.latent_dim, args.latent_dim, bias=False) for modality in args.modalities}

    # load CLIP checkpoint
    saved_state_dict = torch.load(args.checkpoint_to_load, map_location=torch.device('cpu'))
    for modality in args.modalities:
        encoders[modality].load_state_dict(saved_state_dict[f'{modality}_state_dict'])
        if args.projectors:
            projectors[modality].load_state_dict(saved_state_dict[f'projection_matrix_' + modality + '_state_dict'])
            encoder_with_projector = nn.Sequential(encoders[modality], projectors[modality])
            encoders[modality] = encoder_with_projector
        encoders[modality] = encoders[modality].to(device)

    print(f'loaded pretrained encoders from path {args.checkpoint_to_load}')
 
    # store embeddings for different modalities
    embeddings = {modality: [] for modality in args.modalities}

    # declare eval mode
    dummy_models = {modality: nn.Identity() for modality in args.modalities}
    switch_mode(args.modalities, encoders, dummy_models, mode='eval')

    # get embeddings
    with torch.no_grad():
        for data in test_loader:
            # move tensors to device
            data = tensros_to_device(args.modalities, data, device)

            for modality in args.modalities:
                modality_embeddings = encoders[modality](data[modality])
                embeddings[modality].append(modality_embeddings.to('cpu'))

    # concatenate embeddings to form 2D tensors of shape (num_samples, latent_dim)
    embeddings = {modality: torch.cat(embeddings[modality], dim=0) for modality in args.modalities}

    # cross-modality retrieval

    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    top_k_accuracies = {}

    # we want permutations since retieval isn't necessairly symmetric
    modality_pairs = list(itertools.permutations(args.modalities, 2))

    for modality1, modality2 in modality_pairs:
        top1_count = 0
        top5_count = 0
        top10_count = 0
        top25_count = 0
        rank = []
        for idx in range(len(test_dataset)):
            similarity = cosine_similarity(embeddings[modality1][idx:idx+1, :], embeddings[modality2])
            _, indices = torch.topk(similarity, k=len(test_dataset), largest=True, sorted=True)

            # position of idx in indices
            position = (indices == idx).nonzero(as_tuple=True)[0].item() + 1
            
            if position == 1: 
                top1_count += 1
            if position <= 5:
                top5_count += 1
            if position <= 10:
                top10_count += 1
            if position <= 25:
                top25_count += 1
            
            rank.append(position)
        
        top_k_accuracies[f'{modality1}-{modality2}'] = {'top1': top1_count / len(test_dataset), 
                                                        'top5': top5_count / len(test_dataset), 
                                                        'top10': top10_count / len(test_dataset), 
                                                        'top25': top25_count / len(test_dataset),
                                                        'average_rank': np.mean(rank),
                                                        'median_rank': np.median(rank)}
        
    # print results
    print(f'Top-k accuracies and average/median rank for cross-modality retrieval: {top_k_accuracies}')

    # plot retrieval between different modalities
    top_k_categories = ['Top-1', 'Top-5', 'Top-10', 'Top-25']
    topk_acc_dos_crystal = [top_k_accuracies['dos-crystal'][f'top{k}'] for k in [1, 5, 10, 25]]
    topk_acc_charge_density_crystal = [top_k_accuracies['charge_density-crystal'][f'top{k}'] for k in [1, 5, 10, 25]]
    top_k_acc_dos_charge_density = [top_k_accuracies['dos-charge_density'][f'top{k}'] for k in [1, 5, 10, 25]]

    topk_acc_plot = {
        'DOS - Crystal': topk_acc_dos_crystal,
        'Charge Density - Crystal': topk_acc_charge_density_crystal,
        'DOS - Charge Density': top_k_acc_dos_charge_density
    }

    plt.figure(figsize=(5,3))

    for method, accuracies in topk_acc_plot.items():
        plt.plot(top_k_categories, accuracies, marker='o', label=method)

    # Add labels and title
    plt.xlabel('Top-k Accuracies')
    plt.ylabel('Retrieval Accuracy')
    plt.title('Top-k Accuracies for Retrieval')
    plt.legend(fontsize='small')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('retrieval.pdf', dpi=300)
    plt.show()
    

if __name__ == '__main__':
    main()
    print('Done!')
