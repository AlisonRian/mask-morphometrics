import math
from detectron2.engine import DefaultPredictor
from detectron2.data import dataset_mapper
import optuna
from detectron2.data.datasets import register_coco_instances
from trainer import get_train_cfg, train
from split_folds import split_to_holdout_train
import time
import os
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from extract_masks import extract_measurements_from_masks
from detectron2.data import DatasetCatalog
import shutil
import gc
import torch
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_timeline
from utils import evaluate_semantic_metrics
from pycocotools.coco import COCO
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
CLASSES = ['aerial', 'root']
SEED = 42
WORKERS = 4
N_TRIALS = 50
N_STARTUP_TRIALS = 5
DATASET_PATH = "train-maskr-cnn/Dataset_splitted"
FOLDS_DIR = "train-maskr-cnn/folds"
EVAL_PERIOD = 50
def objective(trial, holdout_data, fold):
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    imgsz = trial.suggest_categorical("imgsz", [256, 512, 640])
    iterations = trial.suggest_categorical("iterations", [2600, 3500, 4400])
    degrees = trial.suggest_float("degrees", 0.0, 30.0, step=5.0)
    hsv_v = trial.suggest_float("hsv_v", 0.1, 0.8, step=0.05)
    scale = trial.suggest_float("scale", 0.1, 0.8, step=0.05)
    fliplr = trial.suggest_float("fliplr", 0.0, 0.5, step=0.1)

    #weight_decay = trial.suggest_categorical("weight_decay",[1e-5, 1e-4, 1e-3])
    #lr0 = trial.suggest_categorical("lr0", [0.0001, 0.0005, 0.001])
    #iterations = trial.suggest_categorical("iterations", [2600, 3500, 4400])
    #degrees = trial.suggest_categorical("degrees", [15,25,35])
    #hsv_v = trial.suggest_categorical("hsv_v", [0.4, 0.6, 0.8])
    #fliplr = trial.suggest_categorical("fliplr", [0.0, 0.5, 0.8])
    #scale = trial.suggest_categorical("scale", [0.1, 0.5, 0.8])
    

    if imgsz <= 256:
        dynamic_batch_size = 4
    else:
        dynamic_batch_size = 2

    # 1. Cada trial treina em uma pasta identificada pelo seu número
    output_dir = os.path.join(FOLDS_DIR, f"fold{fold}", f"trial_{trial.number}")
    best_trial_dir = os.path.join(FOLDS_DIR, f"fold{fold}", "best_trial")

    TRAIN_NAME = f"holdout_train_{fold}"
    VAL_NAME = f"holdout_val_{fold}"


    if TRAIN_NAME not in DatasetCatalog.list():
        register_coco_instances(TRAIN_NAME, {}, holdout_data['TRAIN_JSON_DIR'], holdout_data['TRAIN_DATA_DIR'])
    if VAL_NAME not in DatasetCatalog.list():
        register_coco_instances(VAL_NAME, {}, holdout_data['VAL_JSON_DIR'], holdout_data['VAL_DATA_DIR'])

    cfg = get_train_cfg(
        model_name=MODEL_NAME,
        train_name=TRAIN_NAME,
        val_name=VAL_NAME,
        min_size_train=imgsz,
        max_size_train=imgsz,
        eval_period=0,
        num_workers=WORKERS,
        batch_size=dynamic_batch_size,
        learning_rate=lr0,
        iterations=iterations,
        weight_decay=weight_decay,
        amp_enabled=True,
        # solver_steps=(1890, 2430),
        checkpoint_period=450,
        num_classes=2,
        output_dir=output_dir,
        random_seed=SEED,
        degrees=degrees,
        brightness=hsv_v,
        random_flip='horizontal',
        random_flip_prob=fliplr,
        scale = scale
    )
    predictor = None
    coco = None
    try:
        train(output_dir=output_dir, cfg=cfg, resume=False, trial=trial, eval_period=EVAL_PERIOD)
        torch.cuda.empty_cache()
        cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    #evaluator = COCOEvaluator(VAL_NAME, tasks=("bbox", "segm"), output_dir=os.path.join(output_dir, 'optuna_eval'))
    #val_loader = build_detection_test_loader(cfg, VAL_NAME)
    #metrics_dict = inference_on_dataset(predictor.model, val_loader, evaluator)

        predictor = DefaultPredictor(cfg)
        coco = COCO(holdout_data['VAL_JSON_DIR'])
        seg_metrics_dict = evaluate_semantic_metrics(predictor, coco, holdout_data['VAL_DATA_DIR'], VAL_NAME)

        miou = seg_metrics_dict['mIoU']
        mdice = seg_metrics_dict['mDice']
        precision = seg_metrics_dict['Precision']
        recall = seg_metrics_dict['Recall']
        iou_per_cls = seg_metrics_dict['mIoU_per_class']
        dice_per_cls = seg_metrics_dict['mDice_per_class']
        precision_per_cls = seg_metrics_dict['Precision_per_class']
        recall_per_cls = seg_metrics_dict['Recall_per_class']

        metrics_data = {
            'mIoU': miou,
            'mDice': mdice,
            'Precision_pixel': precision,
            'Recall_pixel': recall,
            'Iou_aerial': iou_per_cls['aerial'],
            'Iou_root': iou_per_cls['root'],
            'Dice_aerial': dice_per_cls['aerial'],
            'Dice_root': dice_per_cls['root'],
            'Precision_aerial': precision_per_cls['aerial'],
            'Precision_root': precision_per_cls['root'],
            'Recall_aerial': recall_per_cls['aerial'],
            'Recall_root': recall_per_cls['root'],
        }
        metrics_df = pd.DataFrame([metrics_data])
        metrics_csv = metrics_df.to_csv(index=False, float_format='%.4f')
        with open(os.path.join(output_dir, "segmentation_metrics.csv"), "w", encoding="utf-8") as f:
            f.write(metrics_csv)
        current_miou = seg_metrics_dict["mIoU"]
        if current_miou is None or math.isnan(float(current_miou)):
            current_miou = 0.0

        # 2. Descobre qual era o melhor valor até este trial
        try:
            best_value_so_far = trial.study.best_value
        except ValueError:
            # Se for o primeiro trial (trial 0), ainda não existe best_value
            best_value_so_far = -1.0

        
        if current_miou >= best_value_so_far:
            if os.path.exists(best_trial_dir):
                shutil.rmtree(best_trial_dir)
            shutil.copytree(output_dir, best_trial_dir)
            print(f"Novo melhor trial ({trial.number}) com mIoU = {current_miou:.4f}. Pesos e métricas preservados!")

        
        
        return current_miou
    except optuna.TrialPruned:
        raise
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        if predictor is not None:
            del predictor
        if coco is not None:
            del coco

        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    start = time.time()
    batches = [f"L{i}" for i in range(1, 9)]
    for fold in range(1,9):
        #dataset_path = r"dataset"
        test_batch = batches[fold-1]
        train_batches = [batch for batch in batches if batch != test_batch]
        print(f"\n--- INICIANDO FOLD {fold} Lote de Teste: {test_batch}")

        output_dir = os.path.join(FOLDS_DIR, f"fold{fold}", "train")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        study = optuna.create_study(
            direction="maximize",
            study_name="fold_1",
            storage="sqlite:///optuna.db",
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=N_STARTUP_TRIALS,
                n_warmup_steps=500,
                interval_steps=EVAL_PERIOD
            ),
            load_if_exists=True
        )

        if fold == 1:
            holdout_data = split_to_holdout_train(
                dataset_path=DATASET_PATH,
                train_batches=train_batches,
                fold=fold
            )

            study.optimize(
                lambda trial: objective(trial, holdout_data=holdout_data, fold=fold),
                n_trials=N_TRIALS,
                catch=(Exception,)
            )

            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            print(f"\n--- RELATÓRIO DO TESTE ---")
            print(f"Total de trials completas: {len(complete_trials)}")
            print(f"Total de trials podadas pelo Pruner: {len(pruned_trials)}")
            for t in pruned_trials:
                if t.intermediate_values:
                    step = max(t.intermediate_values.keys())
                    score = t.intermediate_values[step]
                    print(f"  -> Trial {t.number} podada na época {step} com score {score:.4f}")
                else:
                    print(f"  -> Trial {t.number} podada sem passos intermediários.")

        # Treino final com os melhores parâmetros encontrados
        print(f"Melhores parâmetros encontrados para o Fold {fold}:", study.best_params)

        best_params = study.best_params

        if best_params["imgsz"] <= 256:
            dynamic_batch_size = 4
        else:
            dynamic_batch_size = 2

        # Registrando os lotes de treino individualmente para o treino final
        for b in train_batches:
            if b not in DatasetCatalog.list():
                json_path = os.path.join(DATASET_PATH, b, "annotations.json")
                image_dir = os.path.join(DATASET_PATH, b)
                register_coco_instances(b, {}, json_path, image_dir)

        cfg = get_train_cfg(
            model_name=MODEL_NAME,
            train_name=train_batches,
            val_name=[], # Sem conjunto de validação para garantir que não acontecerá data leakage.
            min_size_train=best_params["imgsz"],
            max_size_train=best_params["imgsz"],
            eval_period=0,
            num_workers=WORKERS,
            batch_size=dynamic_batch_size,
            learning_rate=best_params["lr0"],
            iterations=best_params["iterations"],
            weight_decay=best_params["weight_decay"],
            amp_enabled=True,
            # solver_steps=(1890, 2430),
            checkpoint_period=450,
            num_classes=2,
            output_dir=output_dir,
            random_seed=SEED,
            degrees=best_params["degrees"],
            brightness=best_params["hsv_v"],
            random_flip='horizontal',
            random_flip_prob=best_params["fliplr"],
            scale = best_params["scale"]
        )

        train(output_dir=output_dir, cfg=cfg, resume=False)

        model_path = os.path.join(output_dir, "model_final.pth")
        cfg_path = os.path.join(output_dir, 'IS_cfg.pickle')

        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        #cfg.MODEL.DEVICE = 'cuda'

        TEST_NAME = f"test_data_{fold}"
        TEST_DATA_DIR = os.path.join(DATASET_PATH, f"L{fold}")
        TEST_JSON_DIR = os.path.join(DATASET_PATH, f"L{fold}", "annotations.json")

        if TEST_NAME not in DatasetCatalog.list():
            register_coco_instances(TEST_NAME, {}, TEST_JSON_DIR, TEST_DATA_DIR)

        evaluator = COCOEvaluator(TEST_NAME, tasks=("bbox", "segm"), output_dir=os.path.join(output_dir, 'output_eval'))
        test_loader = build_detection_test_loader(cfg, TEST_NAME)
        predictor = DefaultPredictor(cfg)
        metrics_dict = inference_on_dataset(predictor.model, test_loader, evaluator)
        coco = COCO(TEST_JSON_DIR)
        seg_metrics_dict = evaluate_semantic_metrics(predictor, coco, TEST_DATA_DIR, TEST_NAME)

        miou = seg_metrics_dict['mIoU']
        mdice = seg_metrics_dict['mDice']
        precision = seg_metrics_dict['Precision']
        recall = seg_metrics_dict['Recall']
        iou_per_cls = seg_metrics_dict['mIoU_per_class']
        dice_per_cls = seg_metrics_dict['mDice_per_class']
        precision_per_cls = seg_metrics_dict['Precision_per_class']
        recall_per_cls = seg_metrics_dict['Recall_per_class']

        metrics_data = {
            'mIoU': miou,
            'mDice': mdice,
            'Precision_pixel': precision,
            'Recall_pixel': recall,
            'Iou_aerial': iou_per_cls['aerial'],
            'Iou_root': iou_per_cls['root'],
            'Dice_aerial': dice_per_cls['aerial'],
            'Dice_root': dice_per_cls['root'],
            'Precision_aerial': precision_per_cls['aerial'],
            'Precision_root': precision_per_cls['root'],
            'Recall_aerial': recall_per_cls['aerial'],
            'Recall_root': recall_per_cls['root'],
        }
        metrics_df = pd.DataFrame([metrics_data])
        metrics_csv = metrics_df.to_csv(index=False, float_format='%.4f')
        with open(os.path.join(output_dir, "segmentation_metrics.csv"), "w", encoding="utf-8") as f:
            f.write(metrics_csv)

        extract_measurements_from_masks(
            cfg_path=cfg_path,
            model_dir=output_dir,
            output_dir=output_dir,
            images_path=os.path.join(DATASET_PATH, test_batch),
            image_size=best_params["imgsz"],
            fold=fold,
            threshold_test=0.5
        )

        if fold == 1:
            optuna_path = os.path.join(FOLDS_DIR, "optuna")
            os.makedirs(optuna_path, exist_ok=True)
            # Salva o dataframe de estudo e gráficos apenas no fold 1
            study_df = study.trials_dataframe()
            study_df.to_csv(
                os.path.join(optuna_path,"optuna_trials.csv"),
                index=False
            )

            with open(os.path.join("folds", "best_params.txt"), "w") as f:
                f.write(f"Best Value: {study.best_value}\n")
                f.write(str(study.best_params))
            # fig = plot_param_importances(study)
            # fig.write_image(os.path.join(optuna_path,"param_importances.png"))
            # fig.write_html(os.path.join(optuna_path,"param_importances.html"))

            # fig = plot_optimization_history(study)
            # fig.write_image(os.path.join(optuna_path,"optimization_history.png"))
            # fig.write_html(os.path.join(optuna_path,"optimization_history.html"))
            
            # fig = plot_timeline(study)
            # fig.write_image(os.path.join(optuna_path,"timeline.png"))
            # fig.write_html(os.path.join(optuna_path,"timeline.html"))
            
            plots = [
                ("param_importances", plot_param_importances),
                ("optimization_history", plot_optimization_history),
                ("timeline", plot_timeline),
            ]
            for plot_name, plot_func in plots:
                try:
                    fig = plot_func(study)
                    fig.write_html(os.path.join(optuna_path, f"{plot_name}.html"))
                    try:
                        fig.write_image(os.path.join(optuna_path, f"{plot_name}.png"))
                    except Exception as img_err:
                        print(f"Aviso: Não foi possível gerar {plot_name}.png (necessário Chrome): {img_err}")
                except Exception as plot_err:
                    print(f"Aviso: Não foi possível gerar o gráfico {plot_name}: {plot_err}")

        print(f"--- FOLD {fold} FINALIZADO ---")
        # Liberar a memória da GPU usada durante o fold atual
        del cfg, test_loader, evaluator, metrics_dict, predictor, coco
        gc.collect()
        torch.cuda.empty_cache()

    end = time.time()
    total_time = (end-start)/60
    with open(os.path.join(FOLDS_DIR, "completion_time.txt"), "w") as f:
        f.write(f"Total: {total_time:.2f} minutes")