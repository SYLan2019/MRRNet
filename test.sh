# export CUDA_VISIBLE_DEVICES=$1

python test.py --gpus 1 --model sparnet --name MRRNet_m6d16_10middleBlock_SEblock_allattention \
    --load_size 128 --dataset_name single --dataroot test_dirs/CelebA_test_DIC/LR \
    --pretrain_model_path ./check_points/MRRNet_m6d16_10middleBlock_SEblock_allattention/latest_net_G.pth \
    --save_as_dir results_CelebA/MRRNet_m6d16_10middleBlock_SEblock_allattention/

