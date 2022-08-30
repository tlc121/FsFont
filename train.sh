#export CUDA_VISIBLE_DEVICES=4
python3 train.py \
    test \
    cfgs/custom.yaml \
    #--resume \path\to\pretrain.pdparams
