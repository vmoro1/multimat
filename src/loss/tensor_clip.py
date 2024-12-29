import torch
from torch.nn import functional as F

# distributed
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
)


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

def TensorCLIPLoss(embeddings, mask_neg=False, distribute=False, temperature=0.2):
    assert 'crystal' in embeddings.keys()       # crystal must be present since it's the anchor 
    # print("using tensor clip with temperature: ", temperature)
    z1 = torch.nn.functional.normalize(embeddings['crystal'], dim=1)
    z2 = torch.nn.functional.normalize(embeddings['dos'], dim=1)
    z3 = torch.nn.functional.normalize(embeddings['charge_density'], dim=1)
    if distribute:
        z1 = gather_from_all(z1)
        z2 = gather_from_all(z2)
        z3 = gather_from_all(z3)
    
    loss = 0
    b = z1.shape[0]
    mask = torch.zeros([b]*3).cuda()
    mask.fill_diagonal_(1) # this is the hyper diagonal
    flatten_mask = mask.flatten(start_dim=-2,end_dim=-1)
    neg_mask = torch.ones([b]*3).cuda()
    if mask_neg:
        for i in range(b):
            neg_mask[i, i, :] = 0.
            neg_mask[i, :, i] = 0.
        neg_mask.diagonal(dim1=2,dim2=1).copy_(neg_mask.diagonal(dim1=2,dim2=1).subtract_(1.))
        neg_mask.fill_diagonal_(1)
    logits = torch.einsum('ib,jb,kb -> ijk', z1,z2,z3) # this is BxBxB
    logits /= temperature 
    exp_logits = torch.exp(logits) * neg_mask
    # first axis
    logits1 = logits.flatten(start_dim=-2,end_dim=-1)
    exp_logits1 = exp_logits.flatten(start_dim=-2,end_dim=-1)
    log_prob1 = -logits1 + torch.log(exp_logits1.sum(dim=1, keepdim=True))
    loss += (flatten_mask * log_prob1).sum(1).mean()
     # second axis
    logits2 = logits.transpose(0,1).flatten(start_dim=-2,end_dim=-1)
    exp_logits2 = exp_logits.transpose(0,1).flatten(start_dim=-2,end_dim=-1)
    log_prob2 = -logits2 + torch.log(exp_logits2.sum(dim=1, keepdim=True))
    loss += (flatten_mask * log_prob2).sum(1).mean()
     # third axis
    logits3 = logits.transpose(0,2).flatten(start_dim=-2,end_dim=-1)
    exp_logits3 = exp_logits.transpose(0,2).flatten(start_dim=-2,end_dim=-1)
    log_prob3 = -logits3 + torch.log(exp_logits3.sum(dim=1, keepdim=True))
    loss += (flatten_mask * log_prob3).sum(1).mean()
    
    return loss

def create_half_mask(x):
    # hyper diagonals will be 1, semi hyper diagonals will be 0.5, rest will be 0
    n, m, l = x.shape
    assert n == m 
    assert n == l
    mask = torch.zeros([n]*3)
    for i in range(n):
        mask[i, i, :] = 0.5
        mask[i, :, i] = 0.5
    mask.diagonal(dim1=2,dim2=1).copy_(mask.diagonal(dim1=2,dim2=1).add_(0.5))
    return mask

def TensorCLIPLoss2(embeddings, distribute=False, temperature=0.2, mask_non_int=None):
    z1 = torch.nn.functional.normalize(embeddings['crystal'], dim=1)
    z2 = torch.nn.functional.normalize(embeddings['dos'], dim=1)
    z3 = torch.nn.functional.normalize(embeddings['charge_density'], dim=1)

    if distribute:
        z1 = gather_from_all(z1)
        z2 = gather_from_all(z2)
        z3 = gather_from_all(z3)
    
    loss = 0
    b = z1.shape[0]
    ## NOTE THE ORDER IS crystal, dos, charge_density. THIS IS IMPORTANT SINCE MASK IS ALSO DEFINED THIS WAY
    logits = torch.einsum('ib,jb,kb -> ijk', z1,z2,z3) # this is BxBxB
    logits /= temperature 
    M = create_half_mask(logits).cuda() # entries are 1 if all 3 match, 0.5 if 2 match and 0 if none match
    neg_mask = neg_mask = (M == 0).int() # negative examples are made up of only fully mismatch samples
    if mask_non_int is not None:
        assert neg_mask.shape == mask_non_int.shape
        neg_mask = neg_mask * mask_non_int
        M = M * mask_non_int
    exp_logits = torch.exp(logits) * neg_mask
    
    # first axis
    logits1 = logits.flatten(start_dim=-2,end_dim=-1)
    flatten_mask = M.flatten(start_dim=-2,end_dim=-1)
    exp_logits1 = exp_logits.flatten(start_dim=-2,end_dim=-1)
    log_prob1 = -logits1 + torch.log(exp_logits1.sum(dim=1, keepdim=True))
    loss += (flatten_mask * log_prob1).sum(1).mean()
     # second axis
    logits2 = logits.transpose(0,1).flatten(start_dim=-2,end_dim=-1)
    # flatten_mask2 = M.transpose(0,1).flatten(start_dim=-2,end_dim=-1) # all are the same
    exp_logits2 = exp_logits.transpose(0,1).flatten(start_dim=-2,end_dim=-1)
    log_prob2 = -logits2 + torch.log(exp_logits2.sum(dim=1, keepdim=True))
    loss += (flatten_mask * log_prob2).sum(1).mean()
     # third axis
    logits3 = logits.transpose(0,2).flatten(start_dim=-2,end_dim=-1)
    exp_logits3 = exp_logits.transpose(0,2).flatten(start_dim=-2,end_dim=-1)
    log_prob3 = -logits3 + torch.log(exp_logits3.sum(dim=1, keepdim=True))
    loss += (flatten_mask * log_prob3).sum(1).mean()
    
    return loss
    