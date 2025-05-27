is_all=1
is_linear_probe=0
finetune=all
model_type=mfm+register
context_points=512
target_points=96
patch_len=64
stride=64
n_embedding=128
num_slots=8
n_epochs_finetune=20
finetune_percentage=1
freeze_embedding=1
L1_loss=1
is_transfer=1
dset='weather'

# random_seed=2021

for target_points in 96 192 336 720
do
    for dset in 'etth1' 'etth2' 'ettm1' 'ettm2'
    # for dset in 'traffic'
    do
        if [ ! -d "logs" ]; then
        mkdir logs
        fi

        if [ ! -d "logs/LongForecasting" ]; then
            mkdir logs/LongForecasting
        fi
        if [ ! -d "logs/LongForecasting/$model_type" ]; then
            mkdir logs/LongForecasting/$model_type
        fi
        if [ ! -d "logs/LongForecasting/$model_type/$dset" ]; then
            mkdir logs/LongForecasting/$model_type/$dset
        fi

        python -u ROSE_zeroshot.py \
        --is_finetune $is_all \
        --is_linear_probe $is_linear_probe \
        --dset_finetune $dset \
        --context_points $context_points \
        --target_points $target_points \
        --batch_size 64 \
        --num_workers 0\
        --scaler standard \
        --features M\
        --patch_len $patch_len\
        --stride $stride\
        --n_embedding $n_embedding\
        --num_slots $num_slots\
        --L1_loss $L1_loss\
        --revin 1 \
        --n_layers 3\
        --n_heads 16 \
        --d_model 128 \
        --d_ff 512\
        --dropout 0.2\
        --head_drop 0.2 \
        --n_epochs_finetune $n_epochs_finetune\
        --lr 1e-4 \
        --pretrained_model checkpoints_0 \
        --finetuned_model_id 1\
        --model_type $model_type\
        --finetune_percentage $finetune_percentage\
        --one_channel 0\
        --is_transfer $is_transfer\
        --freeze_embedding $freeze_embedding >logs/LongForecasting/$model_type/$dset/$model_type'_zeroshots_'$dset'_cw'$context_points'_tw'$target_points'_patch'$patch_len'stride'$stride'_ne'$n_embedding'_'$finetune'_epoch'$n_epochs_finetune'_is_half'$is_half'_freeze_embedding'$freeze_embedding.log 
    done
done
