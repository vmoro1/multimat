import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .utils import switch_mode, tensros_to_device, tensors_to_cuda
from torch.autograd import Variable


def eval_loop(modalities_encoders, downstream_tasks, encoders, heads, test_loader, gpu, types_of_prediction):
    
    # declare eval
    not_needed = {modality: nn.Identity() for modality in modalities_encoders}
    switch_mode(modalities_encoders, encoders, heads, mode='eval')
    # activation functionsneeded since for some tasks (e.g is_metal) a sigmoid or softmax is combined with the loss function
    # activation_funcs = {'bandgap': nn.Identity(), 'efermi': nn.Identity(), 'eform': nn.Identity(), 'is_metal': nn.Sigmoid()}
    tasks_sigmoid = ['is_metal']
    activation_funcs = {task: nn.Sigmoid() if task in tasks_sigmoid else nn.Identity() for task in downstream_tasks}

    # we need all downstream tasks to have a corresponding activation function
    assert set(downstream_tasks).issubset(set(list(activation_funcs.keys())))

    # dictionary to store targets and predictions for differnt modalities and heads
    store = store = {key1: {key2: {'predictions': torch.tensor(()).cuda(gpu), 'targets': torch.tensor(()).cuda(gpu)} 
                            for key2 in downstream_tasks} for key1 in modalities_encoders}

    with torch.no_grad():
        for data in test_loader:
            modalities_all = modalities_encoders + downstream_tasks
            # data  = tensros_to_device(modalities_all, data, device)
            data  = tensors_to_cuda(modalities_all, data, gpu)
            
            for modality in modalities_encoders:
                for head_task in downstream_tasks:
                    encoder = encoders[modality]
                    head = heads[modality][head_task]
                    activation = activation_funcs[head_task]

                    # forward pass through model
                    embeddings = encoder(data[modality])
                    preds = head(embeddings)
                    preds = activation(preds).flatten()

                    # store predictions and targets
                    store[modality][head_task]['predictions'] = torch.cat((store[modality][head_task]['predictions'], preds))
                    store[modality][head_task]['targets'] = torch.cat((store[modality][head_task]['targets'], data[head_task]))

    # compute metrics
    metrics = {}
    for head_task in downstream_tasks:
        metrics[head_task] = {}
        for modality in modalities_encoders:
            predictions = store[modality][head_task]['predictions']
            targets = store[modality][head_task]['targets']

            if types_of_prediction[head_task] == 'classification':
                predictions = torch.round(predictions)
                accuracy = (torch.round(predictions) == targets).sum().item()
                accuracy = accuracy / len(targets)

                f1 = f1_score(targets.to('cpu'), predictions.to('cpu'))
                metrics[head_task][modality] = {'accuracy': accuracy, 'f1': f1}
            else:
                mse = F.mse_loss(predictions, targets)
                metrics[head_task][modality] = {'mse': mse}

    return metrics


def eval_encoder_decoder(modalities_encoders, 
                         decoder_task, 
                         encoders,
                         decoders, 
                         test_loader, 
                         device, 
                         types_of_prediction,
                         decoder_task_mean,
                         decoder_task_std, crystal_arch='potnet'):

    # declare eval
    not_needed = {modality: nn.Identity() for modality in modalities_encoders}
    switch_mode(modalities_encoders, encoders, decoders, mode='eval')

    # activation functionsneeded since for some tasks (e.g is_metal) a sigmoid or softmax is combined with the loss function
    # activation_funcs = {'bandgap': nn.Identity(), 'efermi': nn.Identity(), 'eform': nn.Identity(), 'is_metal': nn.Sigmoid(), 
    #                     'dos': nn.Identity(), 'dielectric': nn.Identity()}
    tasks_sigmoid = ['is_metal']
    activation_funcs = {task: nn.Sigmoid() if task in tasks_sigmoid else nn.Identity() for task in [decoder_task]}

    # we need decoder task to have a corresponding activation function
    assert set([decoder_task]).issubset(set(list(activation_funcs.keys())))

    # dictionary to store targets and predictions for differnt modalities and heads
    store = {modality: {'predictions': torch.tensor(()).to(device), 'targets': torch.tensor(()).to(device)} 
             for modality in modalities_encoders} 
    with torch.no_grad():
        for data in test_loader:
            # number of samples in batch (could be smaller than batch_size for last batch)
            num_samples_batch = data[decoder_task].shape[0]

            # need to handle cgcnn predictions differently
            if crystal_arch == 'cgcnn':
                for modality in modalities_encoders:
                
                    crystal, target = data[modality], data[decoder_task].cuda(non_blocking=True)
                    decoder = decoders[modality]
                    activation = activation_funcs[decoder_task]
                    
                    crystal_var = (Variable(crystal[0].cuda(non_blocking=True)),
                        Variable(crystal[1].cuda(non_blocking=True)),
                        crystal[2].cuda(non_blocking=True),
                        [crys_idx.cuda(non_blocking=True) for crys_idx in crystal[3]])
                    predictions = activation(decoder(encoders['crystal'](*crystal_var)))
                    targets = target[:,1,:] if decoder_task == 'dos' else target.reshape((num_samples_batch,-1))
                    
                    # store predictions and targets
                    store[modality]['predictions'] = torch.cat((store[modality]['predictions'], predictions), dim=0)
                    store[modality]['targets'] = torch.cat((store[modality]['targets'], targets), dim=0)

            else:
                modalities_all = modalities_encoders + [decoder_task]
                data  = tensros_to_device(modalities_all, data, device)
                for modality in modalities_encoders:
                    encoder = encoders[modality]
                    decoder = decoders[modality]
                    activation = activation_funcs[decoder_task]

                    predictions = activation(decoder(encoder(data[modality])))
                    targets = data[decoder_task][:,1,:] if decoder_task == 'dos' else data[decoder_task].reshape((num_samples_batch,-1))

                    # store predictions and targets
                    store[modality]['predictions'] = torch.cat((store[modality]['predictions'], predictions), dim=0)
                    store[modality]['targets'] = torch.cat((store[modality]['targets'], targets), dim=0)

    # compute metrics
    metrics_encdec = {}
    for modality in modalities_encoders:
        predictions = store[modality]['predictions']
        targets = store[modality]['targets']

        if types_of_prediction[decoder_task] == 'classification':
            predictions = torch.round(predictions)
            accuracy = (torch.round(predictions) == targets).sum().item()
            accuracy = accuracy / len(targets)

            f1 = f1_score(targets.to('cpu'), predictions.to('cpu'))
            metrics_encdec[modality] = {'accuracy': accuracy, 'f1': f1}
        else:
            # unnormalize predictions and targets
            predictions_unnormalized = (predictions * decoder_task_std) + decoder_task_mean
            targets_unnormalized = (targets * decoder_task_std) + decoder_task_mean
            mse = F.mse_loss(predictions_unnormalized, targets_unnormalized)
            mae = F.l1_loss(predictions_unnormalized, targets_unnormalized)

            # relative error (for now we remove the targets which are zero)
            mask = targets_unnormalized != 0
            predictions_wo_zeros = predictions_unnormalized[mask]
            targets_wo_zeros = targets_unnormalized[mask]
            absolute_error = torch.abs(predictions_wo_zeros - targets_wo_zeros)
            relative_error = absolute_error / torch.abs(targets_wo_zeros)
            relative_error = torch.mean(relative_error)

            metrics_encdec[modality] = {'mse': mse, 'mae': mae, 'relative_error': relative_error}

    return metrics_encdec

# def eval_encoder_decoder(modalities_encoders, decoder_task, encoders, decoders, test_loader, gpu, types_of_prediction):

#     # declare eval
#     not_needed = {modality: nn.Identity() for modality in modalities_encoders}
#     switch_mode(modalities_encoders, [], encoders, decoders, {}, mode='eval')

#     # activation functionsneeded since for some tasks (e.g is_metal) a sigmoid or softmax is combined with the loss function
#     # activation_funcs = {'bandgap': nn.Identity(), 'efermi': nn.Identity(), 'eform': nn.Identity(), 'is_metal': nn.Sigmoid(), 
#     #                     'dos': nn.Identity(), 'dielectric': nn.Identity()}
#     tasks_sigmoid = ['is_metal']
#     activation_funcs = {task: nn.Sigmoid() if task in tasks_sigmoid else nn.Identity() for task in [decoder_task]}

#     # we need decoder task to have a corresponding activation function
#     assert set([decoder_task]).issubset(set(list(activation_funcs.keys())))

#     # dictionary to store targets and predictions for differnt modalities and heads
#     store = {modality: {'predictions': torch.tensor(()).cuda(gpu), 'targets': torch.tensor(()).cuda(gpu)} 
#              for modality in modalities_encoders} 
#     with torch.no_grad():
#         for data in test_loader:
#             # number of samples in batch (could be smaller than batch_size for last batch)
#             num_samples_batch = data[decoder_task].shape[0]

#             modalities_all = modalities_encoders + [decoder_task]
#             data  = tensors_to_cuda(modalities_all, data, gpu)
#             for modality in modalities_encoders:
#                 encoder = encoders[modality]
#                 decoder = decoders[modality]
#                 activation = activation_funcs[decoder_task]

#                 predictions = activation(decoder(encoder(data[modality])))
#                 targets = data[decoder_task][:,1,:] if decoder_task == 'dos' else data[decoder_task].reshape((num_samples_batch,-1))

#                 # store predictions and targets
#                 store[modality]['predictions'] = torch.cat((store[modality]['predictions'], predictions), dim=0)
#                 store[modality]['targets'] = torch.cat((store[modality]['targets'], targets), dim=0)

#     # compute metrics
#     metrics_encdec = {}
#     for modality in modalities_encoders:
#         predictions = store[modality]['predictions']
#         targets = store[modality]['targets']

#         if types_of_prediction[decoder_task] == 'classification':
#             predictions = torch.round(predictions)
#             accuracy = (torch.round(predictions) == targets).sum().item()
#             accuracy = accuracy / len(targets)

#             f1 = f1_score(targets.to('cpu'), predictions.to('cpu'))
#             metrics_encdec[modality] = {'accuracy': accuracy, 'f1': f1}
#         else:
#             mse = F.mse_loss(predictions, targets)
#             mae = F.l1_loss(predictions, targets)
#             metrics_encdec[modality] = {'mse': mse, 'mae': mae}

#     return metrics_encdec