# All-pairs CLIP where the normal CLIP loss is used pairwise between all modalities and then added together.

import itertools
import torch
from torch.nn import functional as F
from ..loss.infonce import infoNCE, AllPairsInfoNCE



def all_pairs_clip(embeddings, proj_matrices, args):
    
    loss_pairwise_clip = 0
    modalities_list = list(embeddings.keys())

    # get CLIP loss for all combinations of modalities (not permutations)
    for combination_modalities in itertools.combinations(modalities_list, 2):
        modality_1, modality_2 = combination_modalities
        projection_matrix_1, projection_matrix_2 = proj_matrices[modality_1], proj_matrices[modality_2]
        embedding_1, embedding_2 = embeddings[modality_1], embeddings[modality_2]

        # project and normalize embeddings
        embedding_1 = projection_matrix_1(embedding_1)
        embedding_2 = projection_matrix_2(embedding_2)
        loss = infoNCE(embedding_1, embedding_2, temperature=args.temp, distribute=args.distribute)
        loss_pairwise_clip += loss

    return loss_pairwise_clip

def tri_all_pairs_clip(embeddings, args):
    # anchored clip loss
    loss = AllPairsInfoNCE(embeddings, temperature=args.temp, distribute=args.distribute, mask_dict=args.mask)
    return loss

# def all_pairs_clip(embeddings, proj_matrices, device, args):
    
#     loss_pairwise_clip = 0
#     modalities_list = list(embeddings.keys())

#     # get CLIP loss for all combinations of modalities (not permutations)
#     for combination_modalities in itertools.combinations(modalities_list, 2):
#         modality_1, modality_2 = combination_modalities
#         projection_matrix_1, projection_matrix_2 = proj_matrices[modality_1], proj_matrices[modality_2]
#         embedding_1, embedding_2 = embeddings[modality_1], embeddings[modality_2]

#         # project and normalize embeddings
#         embedding_1 = F.normalize(projection_matrix_1(embedding_1), dim=-1)
#         embedding_2 = F.normalize(projection_matrix_2(embedding_2), dim=-1)

#         logits = (embedding_1 @ embedding_2.T) / args.temp 
#         n = embedding_1.shape[0]
#         labels = torch.arange(0, n, dtype=torch.long).to(device)
#         loss = F.cross_entropy(logits, labels)
#         logitsT = logits.T
#         loss += F.cross_entropy(logitsT, labels)
#         loss /= 2
#         loss_pairwise_clip += loss

#     return loss_pairwise_clip