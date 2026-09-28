import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from detectron2 import config
from math import degrees
import detectron2
import cv2
import numpy as np
import os
import pickle
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import transforms as T
from detectron2.data.transforms import Transform, Augmentation, NoOpTransform
import os

# class GaussianBlurTransform(Transform):
#     """Aplica GaussianBlur com ksize=(0,0) e sigmaX definido em float."""
#     def __init__(self, sigma):
#         super().__init__()
#         self._set_attributes(locals())

#     def apply_image(self, img):
#         # ksize=(0,0) faz o OpenCV calcular o tamanho do kernel a partir do sigma
#         return cv2.GaussianBlur(img, (0, 0), self.sigma)

#     def apply_coords(self, coords):
#         return coords

#     def apply_segmentation(self, segmentation):
#         return segmentation

#     def apply_box(self, box):
#         return box

# class RandomGaussianBlur(Augmentation):
#     """Augmentation de desfoque gaussiano com sigma amostrado de um intervalo."""
#     def __init__(self, prob=0.5, sigma_range=(0.2, 0.8)):
#         super().__init__()
#         self._init(locals())

#     def get_transform(self, image):
#         if np.random.rand() < self.prob:
#             sigma = float(np.random.uniform(self.sigma_range[0], self.sigma_range[1]))
#             return GaussianBlurTransform(sigma)
#         else:
#             return NoOpTransform()

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        # COCOEvaluator calcula AP, AP50, AP75 para bbox e segmentação (se aplicável)
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        target_size = (cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN)
        scale = cfg.INPUT.SCALE

        augmentations = []
        if hasattr(cfg.INPUT, "SCALE") and cfg.INPUT.SCALE > 0.0:
            augmentations.append(
                T.ResizeScale(
                    min_scale=1.0 - scale,
                    max_scale=1.0 + scale,
                    target_height=target_size[0],
                    target_width=target_size[1],
                    interp=2,
                )
            )
            augmentations.append(
                T.FixedSizeCrop(crop_size=target_size, pad=True, pad_value=114.0)
            )
        else:
            augmentations.append(
                T.ResizeShortestEdge(min_size, max_size, sample_style)
            )

        if cfg.INPUT.RANDOM_FLIP == 'horizontal':
            augmentations.append(T.RandomFlip(prob=cfg.INPUT.FLIP_PROB, horizontal=True, vertical=False))
        elif cfg.INPUT.RANDOM_FLIP == "vertical":
            augmentations.append(T.RandomFlip(prob=cfg.INPUT.FLIP_PROB, horizontal=False, vertical=True))
        
        if hasattr(cfg.INPUT, "DEGREES"):
            degrees = cfg.INPUT.DEGREES
            augmentations.append(T.RandomRotation(angle=[-degrees, degrees], expand=False))
        
        if hasattr(cfg.INPUT, "BRIGHTNESS"):
            brightness = cfg.INPUT.BRIGHTNESS
            b_min = max(0.0, 1.0 - brightness)
            b_max = 1.0 + brightness
            augmentations.append(T.RandomBrightness(intensity_min=b_min, intensity_max=b_max))
        
        # if hasattr(cfg.INPUT, "GAUSSIAN_BLUR_PROB"):
        #     blur_prob = cfg.INPUT.GAUSSIAN_BLUR_PROB
        #     blur_sigma = cfg.INPUT.GAUSSIAN_BLUR_SIGMA
        #     augmentations.append(RandomGaussianBlur(prob=blur_prob, sigma_range=blur_sigma))

        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)

def get_train_cfg(
        model_name:str,
        train_name,
        val_name, 
        min_size_train:int, 
        max_size_train:int, 
        eval_period:int, 
        num_workers:int, 
        batch_size:int, 
        learning_rate:float, 
        iterations:int, 
        weight_decay:float, 
        amp_enabled:bool, 
        checkpoint_period:int, 
        num_classes: int,
        output_dir: str,
        degrees: float,
        brightness: float,
        scale: float,
        solver_steps: tuple = (), 
        random_seed: int = 42, 
        random_flip: str = 'horizontal',
        random_flip_prob: float = 0.5,
        ):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

    cfg.DATASETS.TRAIN = tuple(train_name) if isinstance(train_name, (list, tuple)) else (train_name,)
    cfg.DATASETS.TEST = tuple(val_name) if isinstance(val_name, (list, tuple)) else (val_name,)

    cfg.INPUT.MIN_SIZE_TRAIN = (min_size_train,)
    cfg.INPUT.MAX_SIZE_TRAIN = max_size_train
    cfg.INPUT.MIN_SIZE_TEST = min_size_train
    cfg.INPUT.MAX_SIZE_TEST = max_size_train

    cfg.TEST.EVAL_PERIOD = eval_period

    cfg.INPUT.RANDOM_FLIP = random_flip
    cfg.set_new_allowed(True)
    cfg.INPUT.FLIP_PROB = random_flip_prob
    cfg.INPUT.DEGREES = degrees
    cfg.INPUT.BRIGHTNESS = brightness
    cfg.INPUT.SCALE = scale
    cfg.set_new_allowed(False)
    
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.WEIGHT_DECAY = weight_decay

    cfg.SOLVER.STEPS = solver_steps
    #cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.AMP.ENABLED = amp_enabled

    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    # 64 - 0.4 - 400 / 128 - 0.25 - 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.4
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 400
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.MODEL.BACKBONE.FREEZE_AT = 3
    
    cfg.SEED = random_seed

    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = output_dir 

    return cfg

def train(output_dir: str, cfg, resume:bool=True):
    os.makedirs(output_dir, exist_ok=True)
    
    cfg_save_path = os.path.join(output_dir, 'IS_cfg.pickle')
    with open(cfg_save_path, 'wb') as f:
            pickle.dump(cfg,f, protocol=pickle.HIGHEST_PROTOCOL)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

# if __name__ == '__main__':
#     TRAIN_DATA_DIR = r'train-maskr-cnn\dataset\train'
#     TRAIN_JSON_DIR = r'train-maskr-cnn\dataset\train\_annotations_fixed.coco.json'
#     VALID_DATA_DIR = r'train-maskr-cnn\dataset\valid'
#     VALID_JSON_DIR = r'train-maskr-cnn\dataset\valid\_annotations_fixed.coco.json'
#     TRAIN_NAME = "my_dataset_train"
#     VAL_NAME = "my_dataset_val"
#     # Caminho do modelo
#     MODEL_NAME = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#     OUTPUT_DIR = r'train-maskr-cnn\results2'
#     # Definir os dataset
#     # Dataset de Treino
#     register_coco_instances(TRAIN_NAME,{},TRAIN_JSON_DIR, TRAIN_DATA_DIR)
#     # Dataset de Validação
#     register_coco_instances(VAL_NAME,{},VALID_JSON_DIR, VALID_DATA_DIR)
    

#     cfg = get_train_cfg(
#         model_name=MODEL_NAME,
#         train_name=TRAIN_NAME,
#         val_name=VAL_NAME,
#         min_size_train=1024,
#         max_size_train=1024,
#         eval_period=180,
#         num_workers=4,
#         batch_size=2,
#         learning_rate=0.00025,
#         iterations=2700,
#         weight_decay=0.0001,
#         amp_enabled=True,
#         solver_steps=(1890, 2430),
#         checkpoint_period=450,
#         num_classes=2,
#         output_dir=OUTPUT_DIR
#     )

#     train(output_dir=OUTPUT_DIR, cfg=cfg)



    