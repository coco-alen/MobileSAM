# Train

# CUDA_VISIBLE_DEVICES=0 
# python -m torch.distributed.launch --nproc_per_node 4 train.py \

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --model densenet \
#     --expname DenseNet \
#     --bs 16 \
#     --useGPU True \
#     --dataset /data/OpenEDS/OpenEDS/Openedsdata2019/Semantic_Segmentation_Dataset


CUDA_VISIBLE_DEVICES=1 python train.py \
    --model unieye \
    --expname Unieye \
    --bs 16 \
    --useGPU True \
    --dataset /data/OpenEDS/OpenEDS/Openedsdata2019/Semantic_Segmentation_Dataset

# CUDA_VISIBLE_DEVICES=2 python test.py \
# --model densenet \
# --load logs/DenseNet/models/dense_net1.pkl \
# --bs 4 \
# --dataset /data/OpenEDS/Openedsdata2019/Semantic_Segmentation_Dataset/ \
# --save "test"