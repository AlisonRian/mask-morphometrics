from detectron2.engine import HookBase
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import optuna
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
from pycocotools.coco import COCO
from utils import evaluate_semantic_metrics

import torch
import optuna
from detectron2.engine import HookBase
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from pycocotools.coco import COCO
from utils import evaluate_semantic_metrics

class SemanticMetricsHook(HookBase):
    def __init__(self, eval_period: int, trial: optuna.Trial = None):
        self.eval_period = eval_period
        self.trial = trial
        self._coco = None
        self._val_name = None
        self._val_data_dir = None
        self.latest_metrics = {}

    def _build_predictor_wrapper(self):
        cfg = self.trainer.cfg
        model = self.trainer.model
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

        def _predict(image_bgr):
            height, width = image_bgr.shape[:2]
            transformed = aug.get_transform(image_bgr).apply_image(image_bgr)
            tensor_img = torch.as_tensor(transformed.astype("float32").transpose(2, 0, 1)).to(cfg.MODEL.DEVICE)
            inputs = [{"image": tensor_img, "height": height, "width": width}]
            return model(inputs)[0]

        return _predict

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter >= self.trainer.max_iter

        # Executa somente no período configurado ou no final do treino
        if (self.eval_period > 0 and next_iter % self.eval_period == 0) or is_final:
            if self._coco is None:
                self._val_name = self.trainer.cfg.DATASETS.TEST[0]
                meta = MetadataCatalog.get(self._val_name)
                self._val_data_dir = meta.image_root
                self._coco = COCO(meta.json_file)

            # Coloca em modo de inferência
            was_training = self.trainer.model.training
            self.trainer.model.eval()

            try:
                predictor = self._build_predictor_wrapper()
                with torch.no_grad():
                    metrics = evaluate_semantic_metrics(
                        predictor, 
                        self._coco, 
                        self._val_data_dir, 
                        self._val_name
                    )

                self.latest_metrics = metrics
                miou = metrics["mIoU"]

                # 1. Registra no histórico nativo do Detectron2 (Tensorboard / Logs)
                if hasattr(self.trainer, "storage"):
                    self.trainer.storage.put_scalar("val/mIoU", miou)
                    self.trainer.storage.put_scalar("val/mDice", metrics["mDice"])

                # 2. Se estiver rodando dentro do Optuna, reporta e avalia poda
                if self.trial is not None:
                    self.trial.report(miou, step=next_iter)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned(
                            f"Trial {self.trial.number} podado no step {next_iter} (mIoU: {miou:.4f})"
                        )

            finally:
                if was_training:
                    self.trainer.model.train()
            
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
    # Para evitar de realizar a avaliação com COCO_evaluator e realizar apenas com o SemanticHook
    eval_period = 0
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

def train(output_dir: str, cfg, resume:bool=True, trial: optuna.Trial = None, eval_period:int = 0):
    os.makedirs(output_dir, exist_ok=True)
    
    cfg_save_path = os.path.join(output_dir, 'IS_cfg.pickle')
    with open(cfg_save_path, 'wb') as f:
            pickle.dump(cfg,f, protocol=pickle.HIGHEST_PROTOCOL)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=resume)

    if eval_period > 0:
        semantic_hook = SemanticMetricsHook(
            eval_period=eval_period, 
            trial=trial
        )
        trainer.register_hooks([semantic_hook])
    trainer.train()



    