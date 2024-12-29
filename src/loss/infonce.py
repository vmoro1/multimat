import torch
from torch.nn import functional as F

# distributed
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
)
from itertools import combinations

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor

def infoNCE(nn, p, temperature=0.2, distribute=False, mean=False):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    if distribute:
        nn = gather_from_all(nn)
        p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    logitsT = logits.T
    
    if mean:
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        loss += torch.nn.functional.cross_entropy(logitsT, labels, reduction='mean')
    else:
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss += torch.nn.functional.cross_entropy(logitsT, labels)
        
    return loss/2

def TriInfoNCE(embeddings, temperature=0.2, distribute=False, mask_dict=None):
    assert 'crystal' in embeddings.keys()  # crystal must be present since it's the anchor 

    # normalize embeddings
    for mod in embeddings.keys():
        embeddings[mod] = torch.nn.functional.normalize(embeddings[mod], dim=1)
    
    # if distributed setup is used, we gather all the embeddings and mask (if using max data) on a single gpu first
    if distribute:
        for mod in embeddings.keys():
            embeddings[mod] = gather_from_all(embeddings[mod])
        if mask_dict is not None:
            for modmod in mask_dict.keys():
                mask_dict[modmod] = gather_from_all(mask_dict[modmod])
            
    loss = 0
    ii = 0
    for mod in embeddings.keys():
        ii+=1 # count number of pairs
        if mod == 'crystal':
            continue
        z1 = embeddings['crystal']
        z2 = embeddings[mod]
        if mask_dict is not None:
            z1 = z1[mask_dict['crystal_'+mod]]
            z2 = z2[mask_dict['crystal_'+mod]]
            
        logits = z1 @ z2.T
        logits /= temperature
        n = z1.shape[0]
        labels = torch.arange(0, n, dtype=torch.long).cuda()
        logitsT = logits.T
        
        loss += torch.nn.functional.cross_entropy(logits, labels)
        loss += torch.nn.functional.cross_entropy(logitsT, labels)
    
    return loss/(ii*2)  


def AllPairsInfoNCE(embeddings, temperature=0.2, distribute=False, mask_dict=None):
    
    for mod in embeddings.keys():
        embeddings[mod] = torch.nn.functional.normalize(embeddings[mod], dim=1)
        
    if distribute:
        for mod in embeddings.keys():
            embeddings[mod] = gather_from_all(embeddings[mod])
        if mask_dict is not None:
            for modmod in mask_dict.keys():
                mask_dict[modmod] = gather_from_all(mask_dict[modmod])
            
    loss = 0
    ii = 0
    
    for mod1, mod2 in combinations(embeddings.keys(), 2):
        ii+=1
        z1 = embeddings[mod1]
        z2 = embeddings[mod2]
        if mask_dict is not None:
            z1 = z1[mask_dict[f'{mod1}_{mod2}']]
            z2 = z2[mask_dict[f'{mod1}_{mod2}']]

        logits = z1 @ z2.T
        logits /= temperature
        n = z1.shape[0]
        labels = torch.arange(0, n, dtype=torch.long).cuda()
        logitsT = logits.T

        
        loss += torch.nn.functional.cross_entropy(logits, labels)
        loss += torch.nn.functional.cross_entropy(logitsT, labels)
    return loss/(ii*2)