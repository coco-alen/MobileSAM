# Train

# CUDA_VISIBLE_DEVICES=0 
# python -m torch.distributed.launch --nproc_per_node 4 train.py \

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --model densenet \
#     --expname DenseNet_2label \
#     --bs 16 \
#     --useGPU True \
#     --dataset /data/OpenEDS/OpenEDS/Openedsdata2019/Semantic_Segmentation_Dataset


# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --model unieye \
#     --expname Unieye_encorder_trainedKernel_res224 \
#     --bs 16 \
#     --useGPU True \
#     --res 224 \
#     --kernel_path "./logs/kernel_weight.pth" \
#     --dataset /data/OpenEDS/OpenEDS/Openedsdata2019/Semantic_Segmentation_Dataset

CUDA_VISIBLE_DEVICES=2 python test.py \
    --model unieye \
    --expname Unieye_encorder_trainedKernel_res224 \
    --bs 16 \
    --useGPU True \
    --res 224 \
    --kernel_path "./logs/kernel_weight.pth" \
    --dataset /data/OpenEDS/OpenEDS/Openedsdata2019/Semantic_Segmentation_Dataset \
    --load /data/hyou37/MobileSAM/segment/logs/Unieye_encorder_trainedKernel_res224/models/dense_net240.pt \
    --save "test"