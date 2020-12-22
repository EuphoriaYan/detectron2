
export CUDA_VISIBLE_DEVICES=3

python tools/train_net.py \
--config-file \
configs/COCO-Detection/faster_rcnn_R_50_C4_1x_train.yaml \
--num-gpus 1 \
SOLVER.IMS_PER_BATCH 2 \
SOLVER.BASE_LR 0.0025 \
MODEL.ROI_HEADS.NUM_CLASSES 1 \
MODEL.WEIGHTS output/model_final.pth \
