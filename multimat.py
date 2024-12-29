import os
import argparse
from datetime import datetime
import time
import sys
sys.path.append('../')      # needed if script is executed from scripts directory
sys.path.append('../..')    # needed if script is executed from shell directory
sys.path.append('.')        # needed if script is executed from scienceclip directory

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path

from src.model.PotNet.models.potnet import PotNet
from src.model.cgcnn import CrystalGraphConvNet
from src.model.matformer.models.pyg_att import Matformer
from src.model.transformer_dos import TransformerDOS
from src.model.ResNeXt_3D import resnext50
from src.model.text_mlp import TextMLP
from src.data.materials_project.dataset.dataset import MatDataset
from src.utils.utils import (fix_seed, LRScheduler, switch_mode, get_model_params, tensors_to_cuda,
                             get_loss_function_CLIP, adjust_learning_rate, create_decoder, create_mask_to_cuda)
from src.utils.train_eval_utils import eval_loop
from config.matformer_config import matformer_config
from config.potnet_config import potnet_config
from config.cgcnn_config import cgcnn_config
from src.data.materials_project.dataset.collate_functions import collate, collate_cgcnn, collate_mask_non_int, collate_mask_non_int_cgcnn
print("packages import done")

parser = argparse.ArgumentParser(description='MultiMat-training')

# general
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--modalities_encoders', nargs='+', type=str, default=['crystal', 'dos', 'charge_density', 'text'], help='List of modalities')
parser.add_argument('--path_checkpoint', type=str, default='./checkpoints/')
parser.add_argument('--wandb_dir', type=str, default='./wandb_logs')   
parser.add_argument('--wandb_api_key', type=str, default='')   

parser.add_argument('--checkpoint-dir', type=Path, default='./saved_models/',
                        metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=Path , default='./logs/',
                        metavar='LOGDIR', help='path to tensorboard log directory')
# data
parser.add_argument('--train_test_split', action='store_true')
parser.add_argument('--train_fraction', type=float, default=80)
parser.add_argument('--test_fraction', type=float, default=20)

parser.add_argument('--data_path', type=str, default='./data/')

# optimization
parser.add_argument('--batch_size', type=int, default=128) 
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=500) 
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--loss_function_name', type=str, default='anchored_clip', help='anchored_clip or all_pairs_clip or barlow_twins')

# model general
parser.add_argument('--latent_dim', type=int, default=128)

# DOS encoer 
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--dim_head', type=int, default=64)

# Charge density encoder 
parser.add_argument('--data_dim', type=int, default=32)

# 3D ViT
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--depth_vit', type=int, default=4)
parser.add_argument('--heads_vit', type=int, default=8)
parser.add_argument('--dim_head_vit', type=int, default=64)
parser.add_argument('--MLP_dim_vit', type=int, default=512)
parser.add_argument('--pool', type=str, default='mean')
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--distribute', action='store_true'      )
parser.add_argument('--rank', type=int, default=0)
parser.add_argument("--exp", default="test", type=str,
                        help="Name of experiment")
parser.add_argument('--parallel', action='store_true'      )
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--cluster', type=str, default='sc')
parser.add_argument('--log_using', type=str, default='tb', choices=['tb','wandb','none'])

parser.add_argument('--script_name', type=str, default=None)
parser.add_argument('--script_dir', type=str, default=None)
parser.add_argument('--no_eval', action='store_true')
parser.add_argument('--start_from_epoch', type=int, default=0)
parser.add_argument('--crystal_arch', type=str, default='potnet', choices=['matformer','cgcnn','potnet'])
parser.add_argument('--fc_features', type=int, default=256)
parser.add_argument('--lmda', type=float, default=0.0051)
parser.add_argument('--use_proj', action='store_true')
parser.add_argument('--use_final_bn', action='store_true') # set to true for Barlow Twins
parser.add_argument('--barlow_half', action='store_true') # set to true for Barlow Twins
parser.add_argument('--mask_neg', action='store_true') # set to true for Barlow Twins
parser.add_argument('--temp', type=float, default=0.2)
parser.add_argument('--barlow_mask', action='store_true') # set to true for Barlow Twins
parser.add_argument('--text_nonlinear', action='store_true', default=True) # set to true for Barlow Twins
parser.add_argument('--text_depth', type=int, default=3) # set to true for Barlow Twins
parser.add_argument('--mask_non_int', action='store_true') # set to true for Barlow Twins
parser.add_argument('--file_to_keys', type=str, default=None)
parser.add_argument('--file_to_modalities_dicts', type=str, default=None)

def main():
    
    args = parser.parse_args()
    
    local_rank = 0
    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.distribute: # this is for distributed training 
        world_size  = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        print("device count: ", torch._C._cuda_getDeviceCount())
        print("global rank: ", global_rank)
        print("world_size:", world_size)
        
        local_rank  = global_rank % torch._C._cuda_getDeviceCount()

        torch.distributed.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)
        print(f'GPU {global_rank} reporting in. Local rank: {local_rank}. CPU threads: {torch.get_num_threads()}.')
        torch.distributed.barrier()

        if global_rank > 0:
            tqdm = lambda x, total : x
            sys.stdout = open(os.devnull, 'w')
            
        args.rank = global_rank
        args.world_size = world_size
        main_worker(local_rank, args)
    
    else:
        print("Starting single GPU training..")
        main_worker(0, args)

def main_worker(gpu, args):
    
    assert len(args.modalities_encoders) > 1    # For multimat we need at least two modalities
    assert set(args.modalities_encoders).issubset(set(['crystal', 'dos', 'charge_density', 'text']))

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    fix_seed(args.seed)
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.log_using=='tb':
            import tensorboard_logger as tb_logger
            logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)
        elif args.log_using == 'wandb':
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            os.environ["WANDB_MODE"] = 'offline'
            os.environ["WANDB_DIR"] = args.wandb_dir
            wandb.init(
                project="multimat", 
                name=f"{args.exp}")
            
    # dataset
    print("initializing dataset")
    modalities_to_include = args.modalities_encoders 
    
    if args.crystal_arch == 'matformer':
        dataset = MatDataset(modalities=modalities_to_include, data_path=args.data_path, crystal_file='crystal.pt', \
            mask_non_intersect=args.mask_non_int, \
                file_to_keys=args.file_to_keys, file_to_modalities_dicts=args.file_to_modalities_dicts) 
        collate_func = collate_mask_non_int if args.mask_non_int else collate
        
    elif args.crystal_arch == 'cgcnn':
        dataset = MatDataset(modalities=modalities_to_include, data_path=args.data_path, crystal_file='crystal_cgcnn.pt', \
            mask_non_intersect=args.mask_non_int, \
                file_to_keys=args.file_to_keys, file_to_modalities_dicts=args.file_to_modalities_dicts)
        collate_func = collate_mask_non_int_cgcnn if args.mask_non_int else collate_cgcnn
    
    elif args.crystal_arch == 'potnet':
        dataset = MatDataset(modalities=modalities_to_include, data_path=args.data_path, crystal_file='crystal_potnet.pt', \
            mask_non_intersect=args.mask_non_int, \
                file_to_keys=args.file_to_keys, file_to_modalities_dicts=args.file_to_modalities_dicts)
        collate_func = collate_mask_non_int if args.mask_non_int else collate
    
    # get mpids and save them 
    k, md = dataset.get_intersection_keys()
    torch.save(md, args.checkpoint_dir / 'modalities_dict.pt')
    torch.save(k, args.checkpoint_dir / 'keys.pt')
    print(f"Total of {len(k)} materials. mp-ids saved to {args.checkpoint_dir}")
    
    # # no need to create dataset split since all data is "unlabeled"
    if args.train_test_split:
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        total_samples = len(indices)
        train_size = int(args.train_fraction / 100 * total_samples)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        np.save(args.checkpoint_dir / 'train_indices.npy', train_indices)
        np.save(args.checkpoint_dir / 'test_indices.npy', test_indices)
        print("saved train and test indices to npy files")
        train_dataset = torch.utils.data.Subset(dataset, train_indices)

    else:
        train_dataset = dataset
    
    print("dataset intialized!")
    if args.distribute:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=per_device_batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True,
                                                    sampler=train_sampler,
                                                    collate_fn=collate_func)
        
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
            collate_fn=collate_func
        )
    

    crystal_encoder = None
    dos_encoder = None
    charge_density_encoder = None
    text_encoder = None
    print("intializing models")
    if 'crystal' in args.modalities_encoders:
        if args.crystal_arch == 'matformer':
            crystal_encoder = Matformer(matformer_config).cuda(gpu)
            if args.use_proj or args.use_final_bn:
                raise NotImplementedError
        elif args.crystal_arch == 'potnet':
            potnet_config.embedding_dim = args.latent_dim # default is 128
            potnet_config.fc_features = args.fc_features # default is 256
            potnet_config.projector = args.use_proj 
            potnet_config.final_bn = args.use_final_bn
            crystal_encoder = PotNet(potnet_config).cuda(gpu)

        elif args.crystal_arch == 'cgcnn':
            structures = dataset[0]['crystal']
            orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]
            crystal_encoder = CrystalGraphConvNet(orig_atom_fea_len, 
                                            nbr_fea_len,
                                            atom_fea_len=cgcnn_config.atom_fea_len,
                                            n_conv=cgcnn_config.n_conv,
                                            h_fea_len=cgcnn_config.h_fea_len,
                                            n_h=cgcnn_config.n_h,
                                            classification=False).cuda(gpu)
            if args.use_proj or args.use_final_bn:
                raise NotImplementedError
            
    if 'dos' in args.modalities_encoders: 
        dos_encoder = TransformerDOS(dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head, mlp_dim=4 * args.dim, use_final_bn=args.use_final_bn).cuda(gpu)
        if args.use_proj:
            raise NotImplementedError
    if 'charge_density' in args.modalities_encoders:
        charge_density_encoder = resnext50(embedding_dim=args.latent_dim, projector=args.use_proj, batch_norm=args.use_final_bn).cuda(gpu)
    if 'text' in args.modalities_encoders: 
        text_encoder = TextMLP(input_dim=768,latent_dim=args.latent_dim, non_linear=args.text_nonlinear).cuda(gpu)
        print(f"Total text encoder params: {sum(p.numel() for p in text_encoder.parameters())}")
        if args.use_proj:
            raise NotImplementedError
        
        
    if args.distribute:
        if crystal_encoder is not None:
            crystal_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(crystal_encoder)
            crystal_encoder = torch.nn.parallel.DistributedDataParallel(crystal_encoder, device_ids=[gpu],find_unused_parameters=True)
        if dos_encoder is not None:
            dos_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(dos_encoder)
            dos_encoder = torch.nn.parallel.DistributedDataParallel(dos_encoder, device_ids=[gpu])
        if charge_density_encoder is not None:
            charge_density_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(charge_density_encoder)
            charge_density_encoder = torch.nn.parallel.DistributedDataParallel(charge_density_encoder, device_ids=[gpu])
        if dos_decoder is not None:
            dos_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(dos_decoder)
            dos_decoder = torch.nn.parallel.DistributedDataParallel(dos_decoder, device_ids=[gpu])
        if text_encoder is not None:
            text_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(text_encoder)
            text_encoder = torch.nn.parallel.DistributedDataParallel(text_encoder, device_ids=[gpu])
            
    # # dictionary with all encoders
    encoders_all = {'crystal': crystal_encoder, 'dos': dos_encoder, 'charge_density': charge_density_encoder, 'text': text_encoder}

    # dictionary with encoders to be used
    encoders = {key: encoders_all[key] for key in args.modalities_encoders}

    print("models initialized")
    
    # optimization
    # get loss function
    loss_fn = get_loss_function_CLIP(args.loss_function_name)
   
    # get all parameters
    parameters = get_model_params(args.modalities_encoders, encoders, decoders=None)

    optimizer = torch.optim.AdamW(params=parameters,lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    
    lr_scheduler = LRScheduler(
            optimizer=optimizer,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=0,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=0,
            iter_per_epoch=len(train_loader)
        )
        
    scaler = GradScaler()
    
    if args.start_from_epoch > 0 or (args.checkpoint_dir / 'checkpoint.pth').is_file():
        if args.start_from_epoch > 0:
            if not (args.checkpoint_dir / f'checkpoint_epoch{args.start_from_epoch}.pth').is_file():
                raise f"Checkpoint at epoch {args.start_from_epoch} does not exist!"
            ckpt = torch.load(args.checkpoint_dir / f'checkpoint_epoch{args.start_from_epoch}.pth',
                              map_location='cpu')
        elif (args.checkpoint_dir / 'checkpoint.pth').is_file():
            ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                            map_location='cpu')
            
        start_epoch = ckpt['epoch']
        # load state dicts of encoders, projectors and online heads
        for modality in args.modalities_encoders:
            encoders[modality].load_state_dict(ckpt[modality+'_state_dict'])

        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Loaded checkpoint. starting from epoch {start_epoch}")
        # "load" lr scheduler
        for _ in range(start_epoch):
            for _ in range(len(train_loader)):
                lr_scheduler.step()
                
        if args.script_name is not None:
            delay_script = args.script_name[:-3].replace("_delay", "") + '_delay.sh'
            if start_epoch < args.epochs - 1 and os.path.exists(f"{args.script_dir}/{delay_script}"):
                print("delaying script") 
                if args.rank == 0:
                    os.system(f'sbatch {args.script_dir}/{delay_script}')
                
    else:
        start_epoch = 1
        if args.script_name is not None:
            delay_script = args.script_name[:-3].replace("_delay", "") + '_delay.sh'
            if os.path.exists(f"{args.script_dir}/{delay_script}"):
                print("delaying script") 
                if args.rank == 0:
                    os.system(f'sbatch {args.script_dir}/{delay_script}')

    # training
    print("starting training now.. ")
    aft_metrics = time.time()
    
    for e in range(start_epoch, args.epochs + 1):
        
        start = time.time()
        print(f'Time between epoch: {start-aft_metrics:.2f} s')
        
        # declare train
        switch_mode(args.modalities_encoders, encoders, decoders=None, mode='train')
   
        if args.distribute: 
            train_loader.sampler.set_epoch(e)

        # one epoch
        avg_loss = 0
        for step, data_ in enumerate(train_loader, start=(e-1) * len(train_loader)):
            if args.mask_non_int:
                (data, exist_labels) = data_
            else:
                data = data_
            
            lr = adjust_learning_rate(args, optimizer, train_loader, step)
            
            # change tensors to device
            modalities_all = args.modalities_encoders 
            data = tensors_to_cuda(modalities_all, data, gpu)
            if args.mask_non_int:
                args.mask = create_mask_to_cuda(exist_labels, args, gpu)
            else:
                args.mask = None
            # zero grad
            optimizer.zero_grad(set_to_none=True)
            with autocast():

                # get embeddings
                embeddings = {}
                for modality in args.modalities_encoders:
                    z = encoders[modality](data[modality])
                    embeddings[modality] = z
                    if args.crystal_arch == 'cgcnn':
                        raise NotImplementedError # cgcnn predictions need to be handled differently. check run_baseline.py

                # get loss
                loss = loss_fn(embeddings, args)
                   
                # optimization step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # print all losses below and the learning rate
                if args.rank == 0:
                    print(f"Epoch {e:03d} | step {step + 1:03d} / {len(train_loader):03d}")
                    print(f" | CLIP loss: {loss.item():.3f}")

                    if args.log_using == 'tb':
                        logger.log_value('loss', loss.item(), step)
                        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], step)
                    
                    elif args.log_using == 'none':
                        with open(args.checkpoint_dir / 'stats.txt', 'a+') as f:
                            f.write(f"Epoch {e:03d} | step {step + 1:03d} | loss: {loss.item():.3f} \n")
                            
                    avg_loss += loss.item()
        if args.rank == 0 and args.log_using == 'tb':
            logger.log_value('ep_loss', avg_loss/len(train_loader), e)
            logger.log_value('ep_learning_rate', optimizer.param_groups[0]['lr'], e)

        # save checkpoints
        to_save = {}
        
        for modality in args.modalities_encoders:
            # save encoders
            key = modality + '_state_dict'
            to_save[key] = encoders[modality].state_dict()
            
       
        to_save['epoch'] = e
        to_save['optimizer'] = optimizer.state_dict()


        torch.save(to_save, args.checkpoint_dir / 'checkpoint.pth')
        print(f"Saved models to {args.checkpoint_dir / 'checkpoint.pth'}")
        
        if e % 20 == 0 or e == 1:
            torch.save(to_save, args.checkpoint_dir / f'checkpoint_epoch{e}.pth')
            print(f"Saved models to {args.checkpoint_dir / f'checkpoint_epoch{e}.pth'}")


    # close logger
    if args.rank == 0 and args.log_using == 'wandb':
        wandb.finish()

if __name__ == '__main__':
    main()
    print('Done, training completed!')
