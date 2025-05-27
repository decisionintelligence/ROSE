if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ ! -d "logs/LongForecasting" ]; then
    mkdir logs/LongForecasting
fi
if [ ! -d "logs/LongForecasting/pretrain_bigmodel" ]; then
    mkdir logs/LongForecasting/pretrain_bigmodel
fi

model_name='ROSE_mfm_register'
dset_path='pretrain_data'
python -u ROSE_pretrain_all_batch_2task.py \
    --context_points 512 \
    --target_points 720 \
    --batch_size 8192\
    --dset_path $dset_path \
    --num_workers 8\
    --features M\
    --patch_len 64\
    --stride 64\
    --n_embedding 128\
    --revin 1 \
    --n_layers 3\
    --n_heads 16 \
    --d_model 256 \
    --d_ff 512\
    --dropout 0.2\
    --head_drop 0.2 \
    --mask_ratio 0.4\
    --mask_mode freq_multi\
    --mask_nums 4\
    --n_epochs_pretrain 10\
    --pretrained_model_id 1\
    --lr 1e-4 \
    --model_type mfm+register\
    --finetune_percentage 1\
    --is_all 1\
    --model_name $model_name\
    --one_channel 1\
    --is_checkpoints True\
    --checkpoints_freq 1 >logs/LongForecasting/pretrain_bigmodel/Multifreqmask$model_name.log 




