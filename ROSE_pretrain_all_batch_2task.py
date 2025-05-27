

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.ROSE import ROSE
from src.learner_2task_4predicthead import Learner, transfer_weights
from src.callback.tracking_2task_4predicthead import *
from src.callback.patch_mask_2task_predict import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *
from src.data.datamodule import *
import argparse
os.environ['CUDA_VISIBLE_DEVICES']= "4,5,6,7"

seed=2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser ()
# Dataset and dataloadersc
parser.add_argument('--dset_pretrain', type=list, default=['pems03','pems08','solar','fred','nn5'], help='dataset name')
parser.add_argument('--dset_path', type=str, default='data/', help='dataset path')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=32768, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=64, help='patch length')
parser.add_argument('--stride', type=int, default=64, help='stride between patch')
parser.add_argument('--n_embedding', type=int, default=128, help='embedding size')
parser.add_argument('--num_slots', type=int, default=8, help='num_slots')
parser.add_argument('--p', type=float, default=0.5, help='mask loc')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=6, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
parser.add_argument('--mask_mode', type=str, default="freq_multi", help='choice from patch point ')
parser.add_argument('--mask_nums', type=int, default=4, help='choice from patch point ')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=4, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='multi_freq+slot8+vq128_2task', help='for multivariate model or univariate model')
parser.add_argument('--finetune_percentage', type=float, default=0, help='half of the train_set')
parser.add_argument('--is_all', type=int, default=1, help='all of the dataset')
parser.add_argument('--one_channel', type=int, default=1, help='choose 1 channel')
parser.add_argument('--channel_num', type=int, default=0, help='cut random n channel')
parser.add_argument('--model_name', type=str, default='_bigmodel_7_300w_800M_2task_96', help='half of the train_set')
# model save
parser.add_argument('--is_checkpoints', type=bool, default=True, help='save the checkpoints or not')
parser.add_argument('--checkpoints_freq', type=int, default=1, help='the frequency of saving the checkpoints or not')
parser.add_argument('--checkpoints_path', type=str, default="checkpoints/", help='the frequency of saving the checkpoints or not')




args = parser.parse_args()
print('args:', args)
args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+'_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-pretrain' + str(args.n_epochs_pretrain) + '_mask' + str(args.mask_ratio) +'_ne'+ str(args.n_embedding) + '_model' + str(args.pretrained_model_id) + '_mask_mode_' + args.mask_mode +'_'+ args.model_type+ args.model_name
args.save_path = 'saved_models/' + "bigmodel" + '/' + args.model_type + '/'
args.save_checkpoints_path = args.checkpoints_path + args.save_pretrained_model + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
if not os.path.exists(args.save_checkpoints_path): os.makedirs(args.save_checkpoints_path)


# get available GPU devide
set_device()

def get_model(c_in, args):
    """
    c_in: number of variables
    """
    # get number of patches
    num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    print('number of patches:', num_patch)
    
    # get model
    model = ROSE(c_in=c_in,
                target_dim=args.target_points,
                patch_len=args.patch_len,
                stride=args.stride,
                n_embedding=args.n_embedding,
                num_slots=args.num_slots,
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type='pretrain',
                # norm='LayerNorm',
                res_attention=False,
                mask_mode = args.mask_mode,
                mask_nums = args.mask_nums,
                )        
    # weight_path="/home/bigmodel/23_12_17_PatchTST_self_supervised_ts_module/checkpoints/reconstruct/reconstruct_ckp17.pth"
    # model = transfer_weights(weight_path, model, exclude_head=False)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

def find_lr():
    # get dataloader
    dls = DataProviders(args) 
    model= nn.DataParallel(get_model(1, args), device_ids=[0,1,2,3]) 
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(1, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio,maks_mode=args.mask_mode, mask_nums=args.mask_nums)]
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        is_checkpoints = args.is_checkpoints,
                        checkpoints_freq = args.checkpoints_freq,
                        save_checkpoints_path = args.save_checkpoints_path,
                        n_embedding=args.n_embedding,
                        p=args.p,
                        channel_num=args.channel_num
                        )                         
    # fit the data to the model
    # learn.load('/home/bigmoel/checkpoints/patchtst_pretrained_cw512_patch64_stride64_epochs-pretrain100_mask0.4_ne128_model1_mask_mode_freq_multi_multi_freq+slot8+vq128_2task_bigmodel_6_smallmodel_slot8_patch64_monash_2task_96/checkpoints_1.pth',with_opt=True)                

    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def pretrain_func(lr=args.lr):
    # get dataloader list
    dls = DataProviders(args)
    # get model     
    model= nn.DataParallel(get_model(1, args), device_ids=[0,1,2,3]) 
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    
    # get callbacks
    cbs = [RevInCB(1, denorm=False)] if args.revin else []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio,maks_mode=args.mask_mode, mask_nums=args.mask_nums),
         SaveModelCB(monitor='train_loss', fname=args.save_pretrained_model,                       
                        path=args.save_path, with_opt=True)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        is_checkpoints = args.is_checkpoints,
                        checkpoints_freq = args.checkpoints_freq,
                        save_checkpoints_path = args.save_checkpoints_path,
                        n_embedding=args.n_embedding,
                        metrics=[mse],
                        p=args.p,
                        channel_num=args.channel_num
                        ) 
    # fit the data to the model
    ##################
    # learn.load('/home/bigmodel/23_12_17_PatchTST_self_supervised_ts_module/checkpoints/patchtst_pretrained_cw512_patch64_stride64_epochs-pretrain100_mask0.4_ne128_model1_mask_mode_freq_multi_prompt_bigmodel_9_monash_smallmodel_2task_4head_prompt_vq_prompt_cp_cut_3layer/checkpoints_0.pth',with_opt=False)          
    # learn.freeze()
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr,lr_type='Step')  #lr_type: 'OneCycle' 'Step' 'MultiStep' 'Exp' 'Linear'
    
    train_loss = learn.recorder['train_loss']
    train_mse = learn.recorder['train_mse']
    if args.is_all:
        valid_loss = 0
    else:
        valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'train_mse': train_mse, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)


def only_name(path):
  file_name=[]
  a = os.listdir(path)
  for j in a:
    if os.path.splitext(j)[1] == '.csv':
      name = os.path.splitext(j)[0]
      file_name.append(name)
    
  return file_name

if __name__ == '__main__':
    
    # # suggested_lr = find_lr()
    # Pretrain
    # pretrain_func(0.00001)

    finetune_list=['ETTh1','ETTh2','ETTm1','ETTm2','exchange_rate','electricity_hourly_dataset','traffic_hourly_dataset']
    # pretrain_list=only_name('data_1channel')
    # pretrain_list=only_name('monash+testdownsample_1channel')   #注意weather
    # pretrain_list=only_name('all_bigdata')
    
    # cut_list=[]
    # for name in finetune_list:
    #        for item in pretrain_list:
    #             if name in item:
    #                 cut_list.append(item)
    # for name in cut_list:
    #     pretrain_list.remove(name)
    #     print('remove:'+name)
    pretrain_list=only_name(args.dset_path)
    args.dset_pretrain=pretrain_list
    # suggested_lr = find_lr()
    pretrain_func(0.00005)
    print('pretraining completed')

    

