import torch
from torch.nn import functional as F
import numpy as np

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def get_diag_ind(n):
    ind = []
    for i in range(n):
        ind.append(i * n * n + i * n + i)
    return ind

def get_on_off_diagonal(x):
    n, m, l = x.shape
    assert n == m 
    assert n == l
    ind = get_diag_ind(n)
    nind = [ elem for elem in list(np.arange(n*n*n)) if elem not in ind]
    return x.flatten()[ind], x.flatten()[nind]

def create_half_mask(x):
    # hyper diagonals will be 1, semi hyper diagonals will be 0.5, rest will be 0
    n, m, l = x.shape
    assert n == m 
    assert n == l
    mask = torch.zeros([n]*3).cuda()
    for i in range(n):
        mask[i, i, :] = 0.5
        mask[i, :, i] = 0.5
    mask.diagonal(dim1=2,dim2=1).copy_(mask.diagonal(dim1=2,dim2=1).add_(0.5))
    return mask

def create_mask(x):
    n, m, l = x.shape
    assert n == m 
    assert n == l
    mask = torch.zeros([n]*3).cuda()
    mask.fill_diagonal_(1)
    return mask.bool()

def create_4d_mask(x):
    n, m, l, k = x.shape
    assert n == m 
    assert n == l
    assert n == k 
    eye = torch.eye(n, n).cuda()
    shape = [n, n, n, n]
    unsqueezed_eye = eye.unsqueeze(0).unsqueeze(-1)
    a1 = unsqueezed_eye.expand(shape)
    unsqueezed_eye = eye.unsqueeze(0).unsqueeze(0)
    a2 = unsqueezed_eye.expand(shape)
    unsqueezed_eye = eye.unsqueeze(-1).unsqueeze(-1)
    a3 = unsqueezed_eye.expand(shape)
    unsqueezed_eye = eye.unsqueeze(1).unsqueeze(0)
    a4 = unsqueezed_eye.expand(shape)
    unsqueezed_eye = eye.unsqueeze(1).unsqueeze(-1)
    a5 = unsqueezed_eye.expand(shape)
    unsqueezed_eye = eye.unsqueeze(1).unsqueeze(1)
    a6 = unsqueezed_eye.expand(shape)
    mask = (a1 + a2 + a3 + a4 + a5 + a6) 
    return mask

def BTLoss(z1, z2, args, distribute=False, lmda=0.0051):
    # empirical cross-correlation matrix
    # c = self.bn(z1).T @ self.bn(z2)
    c = z1.T @ z2

    # sum the cross-correlation matrix between all gpus
    c.div_(args.batch_size) # this is total batch size
    if distribute:
        torch.distributed.all_reduce(c)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lmda * off_diag
    
    return loss

def BTLoss3D(z1, z2, z3, args, distribute=False, lmda=0.0051):
    # empirical cross-correlation matrix
    # c = self.bn(z1).T @ self.bn(z2)
    c = torch.einsum('ij,ik,il->jkl', z1, z2, z3)
    
    # sum the cross-correlation matrix between all gpus
    c.div_(args.batch_size) # this is total batch size
    if distribute:
        torch.distributed.all_reduce(c)
    on_diag, off_diag = get_on_off_diagonal(c)
    loss = on_diag.add_(-1).pow_(2).sum() + lmda * off_diag.pow_(2).sum()
    
    return loss

def FastBTLoss3D(z1, z2, z3, args, half=False, bt_mask=False, distribute=False, lmda=0.0051):
    # empirical cross-correlation matrix
    # c = self.bn(z1).T @ self.bn(z2)
    c = torch.einsum('ij,ik,il->jkl', z1, z2, z3) 
    # sum the cross-correlation matrix between all gpus
    c.div_(args.batch_size) # this is total batch size
    if distribute:
        torch.distributed.all_reduce(c)
    if half or bt_mask:
        mask = create_half_mask(c)
        if half:
            loss = c[mask==1].add_(-1).pow_(2).sum() + c[mask==0.5].add_(-0.5).pow_(2).sum()
        elif bt_mask:
            loss = c[mask==1].add_(-1).pow_(2).sum() 
        loss += lmda * c[mask==0].pow_(2).sum()
    else:
        mask = create_mask(c)
        loss = c[mask].add_(-1).pow_(2).sum() 
        loss += lmda * c[~mask].pow_(2).sum()
    return loss

def FastBTLoss4D(z1, z2, z3, z4, args, half=False, bt_mask=False, distribute=False, lmda=0.0051):
    # empirical cross-correlation matrix
    # c = self.bn(z1).T @ self.bn(z2)
    c = torch.einsum('ij,ik,il,im->jklm', z1, z2, z3, z4) # this gives DxDxDxD matrix 
    # sum the cross-correlation matrix between all gpus
    c.div_(args.batch_size) # this is total batch size
    if distribute:
        torch.distributed.all_reduce(c)
    if half:
        mask = create_4d_mask(c)
        # loss = ((mask - c)[mask>0]).pow_(2).sum()
        loss = c[mask==6].add_(-1).pow_(2).sum() + c[mask==3].add_(-0.5).pow_(2).sum() + c[mask==2].add_(-1/3).pow_(2).sum() + c[mask==1].add_(-1/6).pow_(2).sum()
        loss += lmda * c[mask==0].pow_(2).sum()
    else:
        raise "4D BT loss only supports half mask config"
    return loss

# remove mask outside of loss
# try removing accessing of elements