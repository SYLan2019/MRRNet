# export CUDA_VISIBLE_DEVICES=$1
# =================================================================================
# Train MRRNet
# =================================================================================

python train.py --gpus 1 --name MRRNet_m7d1_10middleBlock_FCA_allattention --model sparnet \
    --Gnorm "in" --lr 0.0001 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot ../progressiveFSR/crop_celeba --dataset_name celeba --batch_size 12 --total_epochs 20 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500 #--continue_train 


