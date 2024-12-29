# Anchored CLIP loss (where crystal serves as the anchor)

import torch
from torch.nn import functional as F
from ..loss.infonce import infoNCE, TriInfoNCE
from ..loss.barlow_twins import BTLoss, BTLoss3D, FastBTLoss3D, FastBTLoss4D
from ..loss.tensor_clip import TensorCLIPLoss, TensorCLIPLoss2

def anchored_clip(embeddings, args):

    assert 'crystal' in embeddings.keys()       # crystal must be present since it's the anchor 
    crystall_embedding = embeddings['crystal']
   
    # anchored clip loss
    loss_ancored_clip = 0
    for modality in embeddings.keys():
        # crystall embedding is being used as an anchor so we aren't intressted in the case where modality == 'crystal'
        if modality == 'crystal':
            continue
        
        # project and normalize embedding
        modality_embedding = embeddings[modality]
        
        loss = infoNCE(modality_embedding, crystall_embedding, temperature=args.temp, distribute=args.distribute)
        loss_ancored_clip += loss

    return loss_ancored_clip

def tri_anchored_clip(embeddings, args):

    assert 'crystal' in embeddings.keys()       # crystal must be present since it's the anchor 
    crystall_embedding = embeddings['crystal']
    dos_embedding = embeddings['dos']
    chg_embedding = embeddings['charge_density']
    
    # anchored clip loss
    loss = TriInfoNCE(crystall_embedding, dos_embedding, chg_embedding, temperature=args.temp, distribute=args.distribute)
    return loss

def tri_anchored_clip(embeddings, args):
    # anchored clip loss
    loss = TriInfoNCE(embeddings, temperature=args.temp, distribute=args.distribute, mask_dict=args.mask)
    return loss

def barlow_bimodal(embeddings, args):

    assert 'crystal' in embeddings.keys()       # crystal must be present since it's the anchor 
    crystall_embedding = embeddings['crystal']
    
    # anchored clip loss
    loss_ancored_clip = 0
    for modality in embeddings.keys():
        # crystall embedding is being used as an anchor so we aren't intressted in the case where modality == 'crystal'
        if modality == 'crystal':
            continue
        
        # project and normalize embedding
        modality_embedding = embeddings[modality]
        
        loss = BTLoss(modality_embedding, crystall_embedding, args, distribute=args.distribute, lmda=args.lmda)
        loss_ancored_clip += loss

    return loss_ancored_clip

def barlow_trimodal(embeddings, args):

    assert 'crystal' in embeddings.keys()       # crystal must be present since it's the anchor 
    crystall_embedding = embeddings['crystal']
    dos_embedding = embeddings['dos']
    chg_embedding = embeddings['charge_density']

    loss = BTLoss3D(crystall_embedding, dos_embedding, chg_embedding, args, distribute=args.distribute, lmda=args.lmda)

    return loss

def barlow_trimodal_fast(embeddings, args):

    assert 'crystal' in embeddings.keys()       # crystal must be present since it's the anchor 
    crystall_embedding = embeddings['crystal']
    dos_embedding = embeddings['dos']
    chg_embedding = embeddings['charge_density']

    loss = FastBTLoss3D(crystall_embedding, dos_embedding, chg_embedding, args, half=args.barlow_half, bt_mask=args.barlow_mask, distribute=args.distribute, lmda=args.lmda)

    return loss

def barlow_trimodal_crysdostext(embeddings, args):

    assert 'crystal' in embeddings.keys()       # crystal must be present since it's the anchor 
    crystall_embedding = embeddings['crystal']
    dos_embedding = embeddings['dos']
    chg_embedding = embeddings['text']

    loss = FastBTLoss3D(crystall_embedding, dos_embedding, chg_embedding, args, half=args.barlow_half, bt_mask=args.barlow_mask, distribute=args.distribute, lmda=args.lmda)

    return loss

def barlow_quadmodal_fast(embeddings, args):

    assert 'crystal' in embeddings.keys()       # crystal must be present since it's the anchor 
    crystall_embedding = embeddings['crystal']
    dos_embedding = embeddings['dos']
    chg_embedding = embeddings['charge_density']
    text_embedding = embeddings['text']
    loss = FastBTLoss4D(crystall_embedding, dos_embedding, chg_embedding, text_embedding, args, half=args.barlow_half, distribute=args.distribute, lmda=args.lmda)

    return loss

def tensor_clip(embeddings,args):

    loss = TensorCLIPLoss2(embeddings, distribute=args.distribute, temperature=args.temp, mask_non_int=args.mask)
    return loss
