import detectron2
from detectron2.projects import point_rend
import cv2
import numpy as np
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

TRAIN_DATA_DIR = r'C:\Users\RIAN\detectron2\train-maskr-cnn\seedlings.v1i.coco-segmentation\train'
TRAIN_JSON_DIR = r'C:\Users\RIAN\detectron2\train-maskr-cnn\seedlings.v1i.coco-segmentation\train\_annotations.coco.json'
VALID_DATA_DIR = r'C:\Users\RIAN\detectron2\train-maskr-cnn\seedlings.v1i.coco-segmentation\valid'
VALID_JSON_DIR = r'C:\Users\RIAN\detectron2\train-maskr-cnn\seedlings.v1i.coco-segmentation\valid\_annotations.coco.json'
TRAIN_NAME = "my_dataset_train"
VAL_NAME = "my_dataset_val"

register_coco_instances(TRAIN_NAME,{},TRAIN_JSON_DIR, TRAIN_DATA_DIR)
register_coco_instances(VAL_NAME,{},VALID_JSON_DIR, VALID_DATA_DIR)

if __name__ == '__main__':
    cfg = get_cfg()

    point_rend.add_pointrend_config(cfg)

    cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco/164254221/model_final_736f5a.pkl"
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    cfg.DATASETS.TEST = (VAL_NAME,)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 2000
    # cfg.SOLVER.STEPS = (2800,3600)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = r'C:\Users\RIAN\detectron2\train-maskr-cnn\results_point_rend'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    