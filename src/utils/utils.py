import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
from itertools import combinations


# loss
from ..loss.all_pairs_clip import all_pairs_clip, tri_all_pairs_clip
from ..loss.anchored_clip import anchored_clip, tri_anchored_clip, barlow_bimodal, barlow_trimodal, barlow_trimodal_fast, tensor_clip, barlow_trimodal_crysdostext, barlow_quadmodal_fast
from ..loss.low_rank_approx import low_rank_approximation_loss

import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LRScheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, hide_progress=False,
                targets=None):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device=device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def collate(samples):
    """Dataloader helper to batch graphs cross `samples`. `samples` is a list of dictionaries where each dictionary in on 
    the form as per the dataset."""

    dict_to_return = {}
    for modality in samples[0].keys():
        data_moality = [d[modality] for d in samples]

        if modality == 'crystal':
            # batch across graphs
            batched_graphs = Batch.from_data_list(data_moality)
            dict_to_return[modality] = batched_graphs
        else:
            # data modality is a list of torch tensors or a list of floats. Depending on the case we need torch.stack or torch.tensor
            to_stack = True if type(data_moality[0]) == torch.Tensor else False

            if to_stack:
                dict_to_return[modality] = torch.stack(data_moality).float()
            else:
                dict_to_return[modality] = torch.tensor(data_moality).float()
    
    return dict_to_return


def switch_mode(modalities_encoders, encoders, decoders=None, mode='train'):
    """Switch models between train and eval mode. projmats_or_decoders is either the dictionary with projection matrices for CLIP or
    a dictionary with decoders for encpderdecoder.py"""
    for modality in modalities_encoders:
        # print(modality)
        getattr(encoders[modality], mode)()
        if decoders is not None:
            getattr(decoders[modality], mode)()


def get_model_params(modalities_encoders, encoders, decoders=None):
    """"Get parameters of all models. projmats_or_decoders is either the dictionary with projection matrices for CLIP or
    a dictionary with decoders for encpderdecoder.py"""
    parameters = []
    for modality in modalities_encoders:
        parameters += list(encoders[modality].parameters())
        if decoders is not None:
            parameters += list(decoders[modality].parameters())

    return parameters


def tensros_to_device(modalities_all, data, device):
    """change device of tensors"""
    for modality in modalities_all:
        data[modality] = data[modality].to(device, non_blocking=True)

    return data

def tensors_to_cuda(modalities_all, data, gpu):
    """change device of tensors"""
    for modality in modalities_all:
        data[modality] = data[modality].cuda(gpu, non_blocking=True)
    return data

def tensors_to_cuda2(modalities_all, data):
    """change device of tensors"""
    for modality in modalities_all:
        data[modality] = data[modality].cuda()
    return data

def get_loss_function_CLIP(loss_function_name):
    if loss_function_name == 'anchored_clip':
        return anchored_clip
    elif loss_function_name == 'tri_anchored_clip':
        return tri_anchored_clip
    elif loss_function_name == 'all_pairs_clip':
        return all_pairs_clip
    elif loss_function_name == 'tri_all_pairs_clip':
        return tri_all_pairs_clip
    elif loss_function_name == 'barlow_twins':
        return barlow_bimodal
    elif loss_function_name == 'tri_barlow_twins':
        return barlow_trimodal
    elif loss_function_name == 'low_rank_approximation':
        return low_rank_approximation_loss
    elif loss_function_name == 'tri_barlow_twins_fast':
        return barlow_trimodal_fast
    elif loss_function_name == 'tri_barlow_crysdostext':
        return barlow_trimodal_crysdostext
    elif loss_function_name == 'quad_barlow':
        return barlow_quadmodal_fast
    elif loss_function_name == 'tensor_clip':
        return tensor_clip
    else:
        raise ValueError(f'Unknown loss function: {loss_function_name}')
    
def create_decoder(latent_dim, task, output_neurons_per_decoder, tasks_with_ReLU):
    """Create and return decoder or online heads for encoderdecoder.py"""
    if task == 'dos':
        decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, output_neurons_per_decoder['dos'])
        )
        return decoder
    elif task in tasks_with_ReLU:
        return nn.Sequential(nn.Linear(latent_dim, output_neurons_per_decoder[task]), nn.ReLU())
    elif task not in tasks_with_ReLU:
        return nn.Linear(latent_dim, output_neurons_per_decoder[task])
    elif task in tasks_with_ReLU:
        return nn.Sequential(nn.Linear(latent_dim, output_neurons_per_decoder[task]), nn.ReLU())
    elif task not in tasks_with_ReLU:
        return nn.Linear(latent_dim, output_neurons_per_decoder[task])
    else:
        raise ValueError(f'Unknown decoder name: {task}')

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def create_mask_to_cuda(exist_labels, args, gpu):
    """change device of tensors"""
    if args.loss_function_name == 'tri_anchored_clip' or args.loss_function_name == 'tri_all_pairs_clip':
        exist_labels = {key: torch.Tensor(values).bool() for key, values in exist_labels.items()}
        mask = {f'{mod1}_{mod2}': exist_labels[mod1] & exist_labels[mod2] for mod1, mod2 in combinations(exist_labels.keys(), 2)}
        mask = {k:v.cuda(gpu,non_blocking=True) for k,v in mask.items()}

    elif args.loss_function_name == 'tensor_clip':
        # note the order is crystal, dos, charge density
        exist_labels_bool = {key: [bool(value) for value in values] for key, values in exist_labels.items()}
        # Convert lists of booleans to tensors
        exist_labels_tensor = {key: torch.tensor(value) for key, value in exist_labels_bool.items()}
        # Use torch.einsum to perform the logical AND operation
        mask = torch.einsum('i,j,k->ijk', exist_labels_tensor['crystal'], exist_labels_tensor['dos'], exist_labels_tensor['charge_density']).int().cuda(gpu,non_blocking=True)
    return mask