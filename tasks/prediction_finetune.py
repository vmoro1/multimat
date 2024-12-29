
import os
import argparse
from datetime import datetime
import time
import sys
import numpy as np

sys.path.append('.')        # needed if script is executed from scienceclip directory
sys.path.append('../')      # needed if script is executed from scripts directory
sys.path.append('../..')    # needed if script is executed from shell directory

import wandb   

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from torch.autograd import Variable

from src.model.PotNet.models.potnet import PotNet
from src.model.cgcnn import CrystalGraphConvNet
from src.model.matformer.models.pyg_att import Matformer
from src.model.transformer_dos import TransformerDOS
from src.model.ResNeXt_3D import resnext50

from src.data.materials_project.dataset.dataset import MatDataset
from src.utils.utils import (count_parameters, fix_seed, LRScheduler, collate, switch_mode, get_model_params, tensros_to_device, tensors_to_cuda,
                             create_decoder)
from src.utils.train_eval_utils import eval_loop, eval_encoder_decoder
from config.matformer_config import matformer_config
from config.potnet_config import potnet_config
from config.cgcnn_config import cgcnn_config
from src.data.materials_project.dataset.collate_functions import collate, collate_cgcnn

parser = argparse.ArgumentParser(description='encoder-decoder training')

# general
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--modalities_encoders', nargs='+', type=str, default=['crystal'], help='List of modalities for encoders')
parser.add_argument('--decoder_task', type=str, default='bandgap', help='Task for decoder. Can also be dos for encdec pre-training for example')
parser.add_argument('--path_checkpoint', type=str, default='./checkpoints/')
parser.add_argument('--wandb_project_name', type=str, default='scienceclip_ft_dist_experiments')   
parser.add_argument('--wandb_run_name', type=str, default='encoder-decoder', help='Can also be finetuning or end2end for example')
parser.add_argument('--checkpoint_to_finetune', type=str, default=None, help='Path to checkpoint to finetune from')
parser.add_argument('--wandb_dir', type=str, default='./wandb')   
parser.add_argument('--wandb_api_key', type=str, default='')   


# data
parser.add_argument('--train_perc', type=int, default=70)
parser.add_argument('--val_perc', type=int, default=20)
parser.add_argument('--test_fraction', type=float, default=0.1)
parser.add_argument('--split_seed', type=int, default=42)
parser.add_argument('--num_labeled', type=int, default=0)


# optimization
parser.add_argument('--batch_size', type=int, default=128) 
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=200) 
parser.add_argument('--warmup_epochs', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--clip_grad', type=bool, default=False)

# model general
parser.add_argument('--latent_dim', type=int, default=128)

# DOS encoder 
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
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--rank', type=int, default=0)
parser.add_argument("--exp", default="test", type=str,
                        help="Name of experiment")
parser.add_argument('--log_using', type=str, default='tb', choices=['tb','wandb','none'])
parser.add_argument('--checkpoint-dir', type=Path, default='./ft_saved_models/',
                        metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=Path , default='./ft_logs/',
                        metavar='LOGDIR', help='path to tensorboard log directory')
parser.add_argument('--distribute', action='store_true'      )
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--optim', type=str, default='adamw', choices=['adamw','sgd'])

parser.add_argument('--from_pt', action = 'store_true')
parser.add_argument('--eval_only', action = 'store_true')

parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--normalize_targets', action = 'store_true', default=True)
parser.add_argument('--non_normalize_targets', action = 'store_true', default=True)

parser.add_argument('--script_name', type=str, default=None)
parser.add_argument('--script_dir', type=str, default=None)

parser.add_argument('--pt_ckpt', type=str, default=None)
parser.add_argument('--eval_ckpt', type=str, default=None)
parser.add_argument('--no_scheduler', action = 'store_true')
parser.add_argument('--crystal_arch', type=str, default='potnet', choices=['matformer','cgcnn','potnet'])
parser.add_argument('--fc_features', type=int, default=256)
parser.add_argument('--from_nondist', action='store_true')
parser.add_argument('--use_final_bn', action='store_true') # set to true for Barlow Twins

parser.add_argument('--file_to_keys', type=str, default=None)
parser.add_argument('--file_to_modalities_dicts', type=str, default=None)

parser.add_argument('--eval_over_100', action='store_true')
parser.add_argument('--use_old_split', action='store_true')

def main():
    args = parser.parse_args()    
    
    local_rank = 0
    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    if args.distribute: # this is for supercloud distribute 
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
    
def main_worker(gpu,args):

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert set(args.modalities_encoders).issubset(set(['crystal', 'dos', 'charge_density']))
    assert args.decoder_task in ['bandgap', 'eform', 'is_metal', 'efermi', 'dos', 'dielectric', 'dielectric_eig', 
                                 'bulk_modulus', 'shear_modulus', 'elastic_tensor', 'compliance_tensor']
   
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.log_using == 'tb':
            import tensorboard_logger as tb_logger
            logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)
        elif args.log_using == 'wandb':
            
            date = datetime.now()
            date = date.strftime("%Y-%m-%d__%H_%M_%S")
            name_date = f"{args.wandb_run_name}_{date}"

            # set up wandb logger
            config = vars(args)     # convert Namespace object to dictionary and use as config
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            os.environ["WANDB_MODE"] = 'offline'
            os.environ["WANDB_DIR"] = './wandb_logs'
            wandb.init(
                project=args.wandb_project_name, 
                name=name_date,
                entity="scienceclip",
                config=config
                )

            # create checkpoint folder
            os.makedirs(args.path_checkpoint, exist_ok=True)
            os.makedirs(os.path.join(args.path_checkpoint, name_date), exist_ok=True)

    # dataset
    fix_seed(args.seed)
    print("initializing dataset.. ")
    modalities_to_include = list(set(args.modalities_encoders + [args.decoder_task]))
    print("all modalities: ", modalities_to_include)
    
    # dataset = MPDataset(modalities=modalities_to_include, normalize_targets=args.normalize_targets, data_path=args.data_path)   
    if args.crystal_arch == 'matformer':
        dataset = MatDataset(modalities=modalities_to_include, non_normalize_targets=args.non_normalize_targets, data_path=args.data_path, crystal_file='crystal.pt', \
            file_to_keys=args.file_to_keys, file_to_modalities_dicts=args.file_to_modalities_dicts, mask_non_intersect=False)
        collate_func = collate
    elif args.crystal_arch == 'cgcnn':
        dataset = MatDataset(modalities=modalities_to_include, non_normalize_targets=args.non_normalize_targets, data_path=args.data_path, crystal_file='crystal_cgcnn.pt', \
            file_to_keys=args.file_to_keys, file_to_modalities_dicts=args.file_to_modalities_dicts, mask_non_intersect=False)
        collate_func = collate_cgcnn
    elif args.crystal_arch == 'potnet':
        dataset = MatDataset(modalities=modalities_to_include, non_normalize_targets=args.non_normalize_targets, data_path=args.data_path, crystal_file='crystal_potnet.pt', \
            file_to_keys=args.file_to_keys, file_to_modalities_dicts=args.file_to_modalities_dicts, mask_non_intersect=False)
        collate_func = collate
    if not args.non_normalize_targets:
        decoder_task_mean = dataset.mean[args.decoder_task].cuda(gpu)
        decoder_task_std = dataset.std[args.decoder_task].cuda(gpu)
    else:
        # this ensure that unnormalizing doesn't change the targets if we didnät use normalization
        decoder_task_mean = 0
        decoder_task_std = 1
        
    if args.use_old_split:
        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                                        [0.8, 0.0, 0.2])
    else:
    
        snumat = 'snumat_' if 'snumat_data' in args.data_path else ''
        
        if args.file_to_keys is None:
            print(f"saving {len(dataset)} mpid keys to {args.checkpoint_dir}")
            mpid_path = os.path.join(args.checkpoint_dir, f'{snumat}{args.decoder_task}_{len(dataset)}_keys.pt')
            mpid_mod_path = os.path.join(args.checkpoint_dir, f'{snumat}{args.decoder_task}_{len(dataset)}_modalities_dict.pt')
            torch.save(dataset.keys, mpid_path)
            torch.save(dataset.modalities_dicts, mpid_mod_path)
        # if args.file_to_keys is None:
        #     print(f"saving mpid keys to {str(args.checkpoint_dir)}")
        #     mpid_path = (args.checkpoint_dir / f'{args.decoder_task}_{len(dataset)}_keys.pt')
        #     mpid_mod_path = (args.checkpoint_dir / f'{args.decoder_task}_{len(dataset)}_modalities_dict.pt')
        #     torch.save(dataset.keys, mpid_path)
        #     torch.save(dataset.modalities_dicts, mpid_mod_path)
        
        ## create the same test and val and train splits
        
        test_perc = 100 - args.train_perc - args.val_perc
        if args.file_to_keys is not None and 'all_78461' in args.file_to_keys:
            print("Loading npy splits for 78k >>>>>> ")
            npy_train_path = os.path.join(args.data_path, 'train_test_split', f'78k_{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_train.npy')
            npy_val_path = os.path.join(args.data_path, 'train_test_split', f'78k_{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_val.npy')
            npy_test_path = os.path.join(args.data_path, 'train_test_split', f'78k_{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_test.npy')
        else:
            npy_train_path = os.path.join(args.data_path, 'train_test_split', f'{snumat}{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_train.npy')
            npy_val_path = os.path.join(args.data_path, 'train_test_split', f'{snumat}{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_val.npy')
            npy_test_path = os.path.join(args.data_path, 'train_test_split', f'{snumat}{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_test.npy')
        if os.path.exists(npy_train_path):
            train_indices = np.load(npy_train_path)
            val_indices = np.load(npy_val_path)
            test_indices = np.load(npy_test_path)
            print(f"loaded train, val and test indices from npy files in {npy_train_path} etc")
            assert (len(train_indices)+len(val_indices)+len(test_indices) == len(dataset))    

        else:
            os.makedirs(os.path.join(args.data_path, 'train_test_split'), exist_ok=True)
            # Shuffle the indices randomly
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)

            # Calculate the sizes of each split
            total_samples = len(indices)
            train_size, val_size = int(args.train_perc / 100 * total_samples), int(args.val_perc / 100 * total_samples)

            # Split the indices
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            np.save(npy_train_path, train_indices)
            np.save(npy_val_path, val_indices)
            np.save(npy_test_path, test_indices)
            print("saved train, val and test indices to npy files")
            
        
        # if args.num_labeled > 0:
        #     npy_train_path = os.path.join(args.data_path, 'train_test_split', f'{args.decoder_task}_train_{args.num_labeled}.npy')
        #     npy_test_path = os.path.join(args.data_path, 'train_test_split', f'{args.decoder_task}_test_30k.npy')
        #     if os.path.exists(npy_test_path):
        #         test_ind = np.load(npy_test_path)
        #     else:
        #         test_ind = np.random.choice(len(dataset), 30000, replace=False)
        #         np.save(npy_test_path, test_ind)
        #         print(f"saved test indices to {npy_test_path}")
                
        #     if os.path.exists(npy_train_path):
        #         train_ind = np.load(npy_train_path)
        #     else:
        #         non_test_ind = [i for i in range(0,len(dataset)) if i not in test_ind]
        #         train_ind = np.random.choice(non_test_ind, args.num_labeled, replace=False)
        #         np.save(npy_train_path, train_ind)
        #         print(f"saved train indices to {npy_train_path}")
        
        #     train_dataset = torch.utils.data.Subset(dataset, train_ind)
        #     test_dataset = torch.utils.data.Subset(dataset, test_ind)
            
        # else:        
        #     train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, 
        #                                                                     [args.train_fraction, args.validation_fraction, args.test_fraction])
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    print("Train dataset size: ", len(train_dataset))
    print("Val dataset size: ", len(val_dataset))
    print("Test dataset size: ", len(test_dataset))
    
    print("dataset initialized")
    # dataloaders
    if args.distribute:
        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=per_device_batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True,
                                                    # pin_memory=False,
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
    # validation_loader = torch.utils.data.DataLoader(
    #     validation_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     pin_memory=True,
    # )
    test_batch_size = 128 if args.distribute else args.batch_size
    
    tdataset = test_dataset if args.eval_only else val_dataset
    
    test_loader = torch.utils.data.DataLoader(
        tdataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=collate_func
    )

    # define model
    # number of output neurons for the last layer of the different tasks
    output_neurons_per_task = {'bandgap': 1, 'eform': 1, 'efermi': 1, 'is_metal': 1, 'dos': 601, 'dielectric': 9, 'dielectric_eig': 3, 
                               'bulk_modulus': 1, 'shear_modulus': 1, 'elastic_tensor': 15, 'compliance_tensor': 15}
    crystal_encoder = None
    dos_encoder = None
    charge_density_encoder = None
    print("initializing models..")
    if 'crystal' in args.modalities_encoders:
        if args.crystal_arch == 'matformer':
            crystal_encoder = Matformer(matformer_config).cuda(gpu)
        elif args.crystal_arch == 'potnet':
            potnet_config.embedding_dim = args.latent_dim # default is 128
            potnet_config.fc_features = args.fc_features # default is 256
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
    if 'dos' in args.modalities_encoders:
        dos_encoder = TransformerDOS(dim=args.dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head, mlp_dim=4 * args.dim).cuda(gpu)
    if 'charge_density' in args.modalities_encoders:
        charge_density_encoder = resnext50(embedding_dim=args.latent_dim).cuda(gpu)

    
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
    print("models initialized")
    # dictionary with all encoders
    encoders_all = {'crystal': crystal_encoder, 'dos': dos_encoder, 'charge_density': charge_density_encoder}

    # dictionary with encoders to be used
    encoders = {key: encoders_all[key] for key in args.modalities_encoders}
    # get decoders (can either be a linear head for finetuning/end-to-end or a dos decoder for encoder-decoder pretraining)
    # TODO if we normalize the data, we can no longer use ReLU for e.g bandgap or bulk modulus sicne output can be negativ. 
    # Some tasks such as bulk modulus are crucial to normalize
    # tasks_with_ReLU = ['bandgap', 'bulk_modulus', 'shear_modulus', 'elastic_tensor', 'compliance_tensor']
    tasks_with_ReLU = []
    decoders =  {key: create_decoder(args.latent_dim, args.decoder_task, output_neurons_per_task, 
                                     tasks_with_ReLU).cuda(gpu) for key in args.modalities_encoders}
    print("decoders: ", decoders)
    if args.distribute:
        decoders = {key: nn.SyncBatchNorm.convert_sync_batchnorm(decoders[key]) for key in args.modalities_encoders}
        decoders = {key: torch.nn.parallel.DistributedDataParallel(decoders[key], device_ids=[gpu]) for key in args.modalities_encoders}

    if args.from_pt:
        saved_state_dict = torch.load(args.pt_ckpt, map_location=torch.device('cpu'))
        
        for modality in args.modalities_encoders:
            if not args.distribute:
                saved_state_dict[f'{modality}_state_dict'] = {k.replace('module.', ''): v for k, v in saved_state_dict[f'{modality}_state_dict'].items()}
            else:
                if args.from_nondist:
                    saved_state_dict[f'{modality}_state_dict'] = {'module.'+k: v for k, v in saved_state_dict[f'{modality}_state_dict'].items()}
            encoders[modality].load_state_dict(saved_state_dict[f'{modality}_state_dict'])
            encoders[modality] = encoders[modality].cuda(gpu)
        print(f'loaded pretrained encoders from path {args.pt_ckpt}')
        
    elif args.eval_only:
        saved_state_dict = torch.load(args.eval_ckpt, map_location=torch.device('cpu'))
        # load state dicts of encoders, projectors and online heads
        for modality in args.modalities_encoders:
            if not args.distribute:
                saved_state_dict[f'{modality}_state_dict'] = {k.replace('module.', ''): v for k, v in saved_state_dict[f'{modality}_state_dict'].items()}
                saved_state_dict['decoder_'+modality+'_state_dict'] = {k.replace('module.', ''): v for k, v in saved_state_dict['decoder_'+modality+'_state_dict'].items()}
            encoders[modality].load_state_dict(saved_state_dict[modality+'_state_dict'])
            decoders[modality].load_state_dict(saved_state_dict['decoder_'+modality+'_state_dict'])
        print(f"Loaded finetuned encoders decoders and heads from path {args.checkpoint_to_finetune}")


    # track gradients
    if args.log_using == 'wandb':
        count = 1
        for modality in args.modalities_encoders:
            wandb.watch(encoders[modality], log='all', log_freq=100, idx=count)
            wandb.watch(decoders[modality], log='all', log_freq=100, idx=count+1)
            count += 2
            
    # optimization
    # get all parameters
    parameters = get_model_params(args.modalities_encoders, encoders, decoders)
    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(params=parameters,lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    # elif args.optim == 'lars':
    #     optimizer = LARS(parameters, lr=0, weight_decay=args.wd,
    #                 weight_decay_filter=exclude_bias_and_norm,
    #                 lars_adaptation_filter=exclude_bias_and_norm)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(parameters, momentum=0.9,lr=args.lr * args.batch_size / 256,weight_decay=args.wd)
    
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
    
    if (args.checkpoint_dir / 'checkpoint.pth').is_file() and not args.eval_only:
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        # load state dicts of encoders, projectors and online heads
        for modality in args.modalities_encoders:
            encoders[modality].load_state_dict(ckpt[modality+'_state_dict'])
            decoders[modality].load_state_dict(ckpt['decoder_'+modality+'_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Loaded checkpoint. starting from epoch {start_epoch}")
        if not args.no_scheduler:
            for _ in range(start_epoch):
                for _ in range(len(train_loader)):
                    lr_scheduler.step()
        if args.script_name is not None:
            delay_script = args.script_name[:-3].replace("_delay", "") + '_delay.sh'
            if start_epoch < args.epochs - 1 and os.path.exists(f"{args.script_dir}/{delay_script}"):
                print("delaying script") 
                if args.rank == 0: # only submit once
                    os.system(f'sbatch {args.script_dir}/{delay_script}')
    else:
        start_epoch = 1
        if args.script_name is not None:
            delay_script = args.script_name[:-3].replace("_delay", "") + '_delay.sh'
            if os.path.exists(f"{args.script_dir}/{delay_script}"):
                print("delaying script") 
                if args.rank == 0: # only submit once
                    os.system(f'sbatch {args.script_dir}/{delay_script}')

    # training
    best_loss_dict ={}
    save_best = {}
    for modality in args.modalities_encoders:
        best_loss_dict[modality] = 999999
    
    if args.eval_only:
        args.epochs = 1
    for e in range(start_epoch, args.epochs + 1):
        for modality in args.modalities_encoders:
            save_best[modality] = False
        start = time.time()

        # declare train
        if not args.eval_only:
            switch_mode(args.modalities_encoders, encoders, decoders, mode='train')

            if args.distribute:
                train_loader.sampler.set_epoch(e)
            avg_loss = 0
            # one epoch
            # for it, data in enumerate(train_loader):
            
            for step, data in enumerate(train_loader, start = (e-1)*len(train_loader)):
                # number of samples in batch (could be smaller than batch_size for last batch)
                num_samples_batch = data[args.decoder_task].shape[0]

                # zero grad
                optimizer.zero_grad(set_to_none=True)
                
                if args.crystal_arch == 'cgcnn':
                    crystal = data['crystal']
                    targets = data[args.decoder_task].cuda(non_blocking=True)
                    targets = targets.reshape((num_samples_batch,-1))
                    
                    # need to handle cgcnn predictions differently
                    crystal_var = (Variable(crystal[0].cuda(non_blocking=True)),
                        Variable(crystal[1].cuda(non_blocking=True)),
                        crystal[2].cuda(non_blocking=True),
                        [crys_idx.cuda(non_blocking=True) for crys_idx in crystal[3]])

                    
                    with autocast():

                        # get embeddings
                        embeddings = {}
                        for modality in args.modalities_encoders:
                            z = crystal_encoder(*crystal_var)
                            print(z.shape)
                            embeddings[modality] = z
                            
                        
                        all_tasks = [args.decoder_task]
                        classification_tasks = ['is_metal']
                        loss_func_tasks = {task: F.mse_loss if task not in classification_tasks else F.binary_cross_entropy_with_logits for task in all_tasks}

                        loss = 0
                        for modality in args.modalities_encoders:
                            predictions = decoders[modality](embeddings[modality])
                            loss += loss_func_tasks[args.decoder_task](predictions, targets)

                        # optimization step
                        scaler.scale(loss).backward()
                        if args.clip_grad:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1)
                            # nn.utils.clip_grad_value_(parameters, clip_value=1)
                        scaler.step(optimizer)
                        scaler.update()
                        if not args.no_scheduler:
                            lr_scheduler.step()

                    
                else:
                    # crystal = crystal.to(device)
                    # z = crystal_encoder(crystal)
                    # change tensors to device
                    modalities_all = args.modalities_encoders + [args.decoder_task]
                    # data = tensros_to_device(modalities_all, data, device)
                    data = tensors_to_cuda(modalities_all, data, gpu)

                    with autocast():

                        # get embeddings
                        embeddings = {}
                        for modality in args.modalities_encoders:
                            z = encoders[modality](data[modality])
                            embeddings[modality] = z
                            # print(z.shape)
                        # loss functions for different decoders
                        # loss_func_tasks = {'dos': F.mse_loss, 'bandgap': F.mse_loss, 'eform': F.mse_loss, 'efermi': F.mse_loss, 
                        #                    'is_metal': F.binary_cross_entropy_with_logits, 'dielectric': F.mse_loss} 
                        all_tasks = [args.decoder_task]
                        classification_tasks = ['is_metal']
                        loss_func_tasks = {task: F.mse_loss if task not in classification_tasks else F.binary_cross_entropy_with_logits for task in all_tasks}

                        loss = 0
                        for modality in args.modalities_encoders:
                            predictions = decoders[modality](embeddings[modality])
                            targets = data[args.decoder_task][:,1,:] if args.decoder_task == 'dos' else data[args.decoder_task].reshape((num_samples_batch,-1))
                            loss += loss_func_tasks[args.decoder_task](predictions, targets)

                        # optimization step
                        scaler.scale(loss).backward()
                        if args.clip_grad:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1)
                            # nn.utils.clip_grad_value_(parameters, clip_value=1)
                        scaler.step(optimizer)
                        scaler.update()
                        if not args.no_scheduler:
                            lr_scheduler.step()

                if args.rank == 0:
                    # print all losses below and the learning rate
                    print(f"Epoch {e:03d} | It {step + 1:03d} / {len(train_loader):03d}")
                    print(f" | encoder-decoder loss: {loss.item():.3f}")
                    print(f" | Learning Rate {optimizer.param_groups[0]['lr']:.8f}")
                    if args.log_using == 'wandb':
                        # log to wandb 
                        to_log_wandb = {f'heads/{k}/': v for k, v in loss_heads.items()}    # change keys to strucutre logging
                        to_log_wandb['epochs'] = e - 1
                        to_log_wandb['it'] = step
                        to_log_wandb['encdec/encoder-decoder Loss'] = loss.item()
                        to_log_wandb['Learning Rate/learning rate'] = optimizer.param_groups[0]['lr']
                        wandb.log(to_log_wandb)
                    elif args.log_using == 'tb':
                        # for t, dic  in loss_heads.items():
                            # for k, v in dic.items():
                                # logger.log_value(f'heads/{t}_{k}', v.item(), step)
                        logger.log_value('enc-dec_loss', loss.item(), step)
                        logger.log_value('enc-dec learning_rate', optimizer.param_groups[0]['lr'], step)

                    avg_loss += loss.item()
                    
            if args.rank == 0 and args.log_using == 'tb':
                logger.log_value('ep_loss', avg_loss/len(train_loader), e)
                logger.log_value('ep_learning_rate', optimizer.param_groups[0]['lr'], e)
                print(f'Time for epoch: {time.time() - start:.2f} s')

        metrics = None
        metric_encdec = None        
        # evaluation on 1 gpu
        if args.eval_only:
            print("EVALUATING ON TEST SET >>>>>")
            
        if args.eval_over_100 and e < 100:
            pass
            
        elif args.rank == 0 and e % args.eval_freq == 0:
            all_tasks = [args.decoder_task]
            classification_tasks = ['is_metal']
            types_of_prediction = {task: 'classification' if task in classification_tasks else 'regression' for task in all_tasks}
            metric_encdec = eval_encoder_decoder(args.modalities_encoders, args.decoder_task, encoders, decoders, test_loader, gpu, 
                                                types_of_prediction,decoder_task_mean,decoder_task_std, crystal_arch=args.crystal_arch)
            # metric_encdec = eval_encoder_decoder(args.modalities_encoders, args.decoder_task, encoders, decoders, test_loader, device, 
                                                # types_of_prediction)

            # print all metrics
            print(f'Test metrics at epoch {e:03d} / {args.epochs:03d} | time for epoch: {time.time() - start:.2f} s')

            # eval of encoder-decoder
            print('evaluation of encoder-decoder')
            for modality in args.modalities_encoders:
                if types_of_prediction[args.decoder_task] == 'classification':
                    print(f'{modality} {args.decoder_task} | Accuracy: {metric_encdec[modality]["accuracy"]:.4f} | F1-score: {metric_encdec[modality]["f1"]:.4f}') 
                else:
                    line_to_print=f'{modality} {args.decoder_task} | MSE: {metric_encdec[modality]["mse"]:.4f} | MAE: {metric_encdec[modality]["mae"]:.4f}'
                    print(line_to_print)
                if args.eval_only:
                    with open(f'./ft_logs/paper_{args.decoder_task}.txt', 'a+') as f:
                        f.write(args.eval_ckpt + "\n" + line_to_print + "\n\n")
            # save best model on the val set
            for modality in args.modalities_encoders:
                if metric_encdec[modality]["mae"] < best_loss_dict[modality]:
                    best_loss_dict[modality] = metric_encdec[modality]["mae"]
                    save_best[modality] = True
                
        
            if args.eval_only:
                sys.exit()
                
            if args.log_using == 'tb':
                # for t, dic in metrics.items():
                #     for k, dic2 in dic.items():
                #         for m, v in dic2.items():
                #             logger.log_value(f'eval/{t}_{k}_{m}', float(v), e)
                for t, dic in metric_encdec.items():
                    for k, v in dic.items():
                        # for m, v in dic2.items():
                        logger.log_value(f'eval_encdec/{t}_{k}', float(v), e)            
            elif args.log_using == 'wandb':
                # log to wandb
                # metrics_wandb = {f'eval/{k}/': v for k, v in metrics.items()}    # change keys to strucutre logging
                metrics_wandb = {}
                metrics_wandb['epochs'] = e
                # metrics_wandb['it'] = e * len(train_loader)
                metric_encdec_wandb = {f'eval_encdec/{args.decoder_task}/': metric_encdec}
                metrics_wandb.update(metric_encdec_wandb)   # combine dicts
                wandb.log(metrics_wandb)
                
        if args.rank == 0:
            # save checkpoints
            to_save = {}
            for modality in args.modalities_encoders:
                # save encoders
                key = modality + '_state_dict'
                to_save[key] = encoders[modality].state_dict()

                # save decoders
                key = 'decoder_' + modality + '_state_dict'
                to_save[key] = decoders[modality].state_dict()

            
            to_save['epoch'] = e
            to_save['optimizer'] = optimizer.state_dict()
           
            # save metrics
            if metric_encdec is not None:
                # to_save['metrics'] = metrics 
                to_save['metric_encdec'] = metric_encdec

            torch.save(to_save, args.checkpoint_dir / 'checkpoint.pth')
            print(f"Saved models to {args.checkpoint_dir / 'checkpoint.pth'}")
            
            if e % 10 == 0 or e == 1:
                torch.save(to_save, args.checkpoint_dir / f'checkpoint_epoch{e}.pth')
                print(f"Saved models to {args.checkpoint_dir / f'checkpoint_epoch{e}.pth'}")
            
            for modality in args.modalities_encoders:
                if save_best[modality]:
                    torch.save(to_save, args.checkpoint_dir / f'checkpoint_best_val.pth')
                    print(f"Saved current BEST models to {args.checkpoint_dir / f'checkpoint_best_val.pth'}")
    print(">>>> BEST LOSS DICT >>>> ")
    print(best_loss_dict)
    # close logger
    if args.rank == 0 and args.log_using == 'wandb':
        wandb.finish()

if __name__ == '__main__':
    main()
    print('Done, training completed!')