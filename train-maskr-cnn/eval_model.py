from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from detectron2.engine import DefaultPredictor
import os
import pickle
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
import cv2
import numpy as np
from utils import compute_iou_and_dice
# Define o dataset de teste
TEST_NAME = "my_dataset_test"
TEST_DATA_DIR = r'train-maskr-cnn\Dataset_splitted\L1'
TEST_JSON_DIR = r'train-maskr-cnn\Dataset_splitted\L1\annotations.json'
register_coco_instances(TEST_NAME,{},TEST_JSON_DIR, TEST_DATA_DIR)

output_dir = r'train-maskr-cnn\results2'
# Carrega a cfg utilizada no treinamento
cfg_saved_path = os.path.join(output_dir, 'IS_cfg.pickle')
with open(cfg_saved_path, 'rb') as f:
    cfg = pickle.load(f)

# Carrega o modelo treinado

cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
# Define o limiar de confiança na inferência 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = 'cpu'

# Executa a inferência e salva as métricas das Bbox e masks (mAP50, mAP50-95)
# if __name__ == '__main__':
    # Instancia o avaliador COCO
    # evaluator = COCOEvaluator(TEST_NAME, output_dir=os.path.join(output_dir, 'output_eval'))
    # val_loader = build_detection_test_loader(cfg, TEST_NAME)

    # # Executa a inferência em todo o conjunto
    # metrics_dict = inference_on_dataset(DefaultPredictor(cfg).model, val_loader, evaluator)
# if __name__ == '__main__':
#     predictor = DefaultPredictor(cfg)
#     coco = COCO(TEST_JSON_DIR)
    
#     # Chama a função nova (altere per_class=True para ver por classe)
#     seg_metrics_dict = evaluate_semantic_metrics(predictor, coco, TEST_DATA_DIR, TEST_NAME)