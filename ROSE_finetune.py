

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.ROSE_lowrank import ROSE
# from src.models.patchTST import PatchTST
from src.learner_pvq import Learner, transfer_weights
from src.callback.core import *
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import * 

import argparse

os.environ['CUDA_VISIBLE_DEVICES']='4'

seed=2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=1, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
# Dataset and dataloader
parser.add_argument('--dset_finetune', type=str, default='etth1', help='dataset name')
parser.add_argument('--context_points', type=int, default=512, help='sequence length')
parser.add_argument('--target_points', type=int, default=720, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# Patch
parser.add_argument('--patch_len', type=int, default=64, help='patch length')
parser.add_argument('--stride', type=int, default=64, help='stride between patch')
parser.add_argument('--n_embedding', type=int, default=128, help='embedding size')
parser.add_argument('--L1_loss', type=int, default=1, help='use L1_loss')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=256, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=512,help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs_finetune', type=int, default=6, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default='full-shot', help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='mfm+register', help='for multivariate model or univariate model')
parser.add_argument('--finetune_percentage', type=float, default=1, help='percentage of the train_set')
parser.add_argument('--one_channel', type=int, default=0, help='choose 1 channel')
parser.add_argument('--freeze_embedding', type=int, default=1, help='freeze the embedding layer')

args = parser.parse_args()
print('args:', args)
args.save_path = 'saved_models/' + args.dset_finetune + '/' + args.model_type + '/'
args.pretrain_path = 'saved_models/' + 'bigmodel' + '/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)

# args.save_finetuned_model = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) + '_mask' + str(args.mask_ratio)  + '_model' + str(args.finetuned_model_id)
suffix_name = '_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride) + '_epochs-finetune' + str(args.n_epochs_finetune) +'_ne'+ str(args.n_embedding) + '_is_half'+ str(args.finetune_percentage) +'freeze_embedding'+str(args.freeze_embedding)+'_model' + str(args.finetuned_model_id)
if args.is_finetune: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name
elif args.is_linear_probe: args.save_finetuned_model = args.dset_finetune+'_patchtst_linear-probe'+suffix_name
else: args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name

# get available GPU devide
set_device()

def get_model(c_in, args, head_type, weight_path=None):
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
                num_patch=num_patch,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                # norm ='LayerNorm',
                act='relu',
                head_type=head_type,
                res_attention=False
                )    
    if weight_path: model = transfer_weights(weight_path, model,exclude_head=False)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def find_lr(head_type):
    # get dataloader
    dls = get_dls(args)    
    # transfer weight
    weight_path = args.pretrain_path+args.pretrained_model+'.pth'
    model = get_model(dls.vars, args, head_type,weight_path)
    # print(f'model_path = {weight_path}')
    # model = transfer_weights(weight_path, model)
    # get loss
    if args.L1_loss==1:
        loss_func = torch.nn.L1Loss(reduction='mean')
    else:
        loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        n_embedding=args.n_embedding,
                        target_points=args.target_points,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def save_recorders(learn):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_finetuned_model + '_losses.csv', float_format='%.6f', index=False)


def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args)
    # get model 
    # transfer weight
    weight_path = args.pretrain_path+args.pretrained_model+'.pth'
    model = get_model(dls.vars, args, head_type='prediction', weight_path=weight_path )
    # model = transfer_weights(weight_path, model) 
    # get loss
    if args.L1_loss==1:
        loss_func = torch.nn.L1Loss(reduction='mean')
    else:
        loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path),
         EarlyStoppingCB(monitor='valid_loss', patient=30)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse],
                        n_embedding=args.n_embedding,
                        target_points=args.target_points,
                        )                            
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.n_epochs_finetune, base_lr=lr, freeze_epochs=20, freeze_embedding=args.freeze_embedding)
    save_recorders(learn)


def linear_probe_func(lr=args.lr):
    print('linear probing')
    # get dataloader
    dls = get_dls(args)
    # get model 
    # transfer weight
    weight_path = args.pretrain_path+args.pretrained_model+'.pth'
    model = get_model(dls.vars, args, head_type='prediction',weight_path=weight_path )
    # model = transfer_weights(weight_path, model)
    # get loss
    if args.L1_loss==1:
        loss_func = torch.nn.L1Loss(reduction='mean')
    else:
        loss_func = torch.nn.MSELoss(reduction='mean')  
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_finetuned_model, path=args.save_path),
        EarlyStoppingCB(monitor='valid_loss', patient=50)
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        metrics=[mse],
                        n_embedding=args.n_embedding,
                        target_points=args.target_points,
                        )                            
    # fit the data to the model
    learn.linear_probe(n_epochs=args.n_epochs_finetune, base_lr=lr)
    save_recorders(learn)


def test_func(weight_path):
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args, head_type='prediction').to('cuda')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs,n_embedding=args.n_embedding,target_points=args.target_points)
    out  = learn.test(dls.test, weight_path=weight_path+'.pth', scores=[mse,mae])         # out: a list of [pred, targ, score]
    print('score:', out[2])
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc.csv', float_format='%.6f', index=False)
    return out



if __name__ == '__main__':

    mse_list =np.array([])
    mae_list =np.array([])
    for i in range(1):
        if args.is_finetune:
            args.dset = args.dset_finetune
            # Finetune
            suggested_lr = find_lr(head_type='prediction')   
            suggested_lr = args.lr   
            finetune_func(args.lr)        
            print('finetune completed')
            # Test
            out = test_func(args.save_path+args.save_finetuned_model)         
            print('----------- Complete! -----------')
            mse_list = np.append(mse_list,out[2][0])
            mae_list = np.append(mae_list,out[2][1])

        elif args.is_linear_probe:
            args.dset = args.dset_finetune
            # Finetune
            suggested_lr = find_lr(head_type='prediction')        
            linear_probe_func(suggested_lr)        
            print('finetune completed')
            # Test
            out = test_func(args.save_path+args.save_finetuned_model)        
            print('----------- Complete! -----------')

        else:
            args.dset = args.dset_finetune
            weight_path = args.save_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
            # Test
            out = test_func(weight_path)        
            print('----------- Complete! -----------')
            
    print(f'mse_mean:{mse_list.mean()},mae_mean:{mae_list.mean()}')
    result_mean = np.array([mse_list.mean(),mae_list.mean()])
    pd.DataFrame(result_mean.reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_finetuned_model + '_acc_mean.csv', float_format='%.6f', index=False)
        


