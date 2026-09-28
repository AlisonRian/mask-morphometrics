import pickle
import os
import cv2
import math
import numpy as np
import logging
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from utils import bigger_skeleton, calculate_pixel_to_cm_ratio, dim_reduction
import pandas as pd
from natsort import natsorted

import sys

# Configuração básica
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_images(images_path: str, image_size: int) -> tuple[list,list]:
    images = []
    file_names = []
    if os.path.isdir(images_path):
        for i in natsorted(os.listdir(images_path)):
            if not i.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image = cv2.imread(os.path.join(images_path,i))
            images.append(image)

            name = ""
            pos = i.find("_")
            if pos != -1:
                name = i[:pos]
            else:
                logging.warning("Error: Formatação não esperada!")
            file_names.append(name)
    else:
        image = cv2.imread(images_path)
        images.append(image)

        filename = os.path.basename(images_path)
        name = ""
        pos = filename.find("_")
        if pos != -1:
            name = filename[:pos]
        else:
            logging.warning("Error: Formatação não esperada!")
        file_names.append(name)

    return images, file_names

def extract_measurements_from_masks(
    cfg_path: str,
    model_dir: str,
    output_dir: str,
    images_path:str,
    image_size: int,
    fold: int,
    threshold_test: float=0.5,
    ):
    # Carrega a cfg utilizada no treinamento
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)

    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_test
    cfg.MODEL.DEVICE = 'cuda'

    predictor = DefaultPredictor(cfg)
    CLASSES = ['aerial', 'root']
    
    images, filenames = load_images(images_path, image_size)

    dfs = []
    for img, filename in zip(images, filenames):
        # Realiza a inferência na imagem
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        # boxes_list = instances.pred_boxes.tensor.tolist()
        # Ordena as instancias da esquerda para a direita
        # sorted_indexes = sorted(range(len(boxes_list)), key=lambda i: boxes_list[i][0])
        # sorted_instances = instances[sorted_indexes]
        boxes_list = instances.pred_boxes.tensor.tolist()
        masks = instances.pred_masks.numpy()
        classes = instances.pred_classes.numpy()
        # Calcula o fator de escala

        scale = calculate_pixel_to_cm_ratio(img)
        if scale is None:
            print(f"Error: Fator de escala não encontrado em {filename}!")
            continue

        aerials_data = []
        roots_data = []
        
        for i, mask in enumerate(masks):
            # No Detectron2 a máscara vem como booleana (True/False)
            mask_img = (mask * 255).astype(np.uint8) 
            class_id = classes[i]
            x1, y1, x2, y2 = boxes_list[i]
            
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            # Assumindo que 0 é Aerial e 1 é Root (Ajuste se o seu array CLASSES for diferente)
            if class_id == 0:
                aerials_data.append({"mask": mask_img, "cx": cx, "cy": cy})
            else:
                roots_data.append({"mask": mask_img, "cx": cx, "cy": cy})
        # Ordenar da esquerda para a direita (bom para manter a ordem visual no CSV)
        aerials_data.sort(key=lambda item: item["cx"])
        roots_data.sort(key=lambda item: item["cx"])
        matched_masks = []
        used_roots = set()
        # Pareamento guloso (Greedy Matching) pela distância dos centros
        for aerial in aerials_data:
            best_idx = -1
            best_dist = float('inf')
            
            for j, root in enumerate(roots_data):
                if j in used_roots:
                    continue
                
                # Distância Euclidiana
                dist = math.hypot(aerial["cx"] - root["cx"], aerial["cy"] - root["cy"])
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            
            # Se achou uma raiz para essa parte aérea, cria o par
            if best_idx != -1:
                matched_masks.append((aerial["mask"], roots_data[best_idx]["mask"]))
                used_roots.add(best_idx)
            else:
                # Se não achou, pareia com máscara vazia (resultará em 0.0 cm)
                empty_mask = np.zeros_like(aerial["mask"])
                matched_masks.append((aerial["mask"], empty_mask))
        # Adicionar raízes que sobraram sem parte aérea
        for j, root in enumerate(roots_data):
            if j not in used_roots:
                empty_mask = np.zeros_like(root["mask"])
                matched_masks.append((empty_mask, root["mask"]))

        aerials = []
        roots = []
        type_map = {0:'Aerial', 1:'Root'}
        for aerial, root in matched_masks:
            masks = [aerial, root]

            for idx, mask in enumerate(masks):

                if cv2.countNonZero(mask) == 0:
                    logger.warning(f'Nenhuma máscara encontrada para o tipo {type_map[idx]}. Pulando...')
                    if idx == 0: aerials.append(0.0)
                    if idx == 1: roots.append(0.0)
                    continue

                # 3 máscaras das plântulas ordenadas da direita para a esquerda.
                skeleton = cv2.ximgproc.thinning(mask,thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                points_skel = bigger_skeleton(skeleton)

                if not points_skel: 
                    logger.warning(f"Nenhum esqueleto para o tipo {type_map[idx]}. Pulando...")
                    if idx == 0: aerials.append(0.0)
                    if idx == 1: roots.append(0.0)
                    continue

                pixels = 0
                for i in range(len(points_skel)-1):
                    x1, y1 = points_skel[i]
                    x2, y2 = points_skel[i+1]
                    distance = math.hypot(x2-x1,y2-y1)
                    pixels+= distance

                length_cm = pixels/scale
                length_cm = round(length_cm, 2)
                print(f"{type_map[idx]} : {length_cm}")
                
                if idx == 0:
                    aerials.append(length_cm)
                elif idx == 1:
                    roots.append(length_cm)

        splitted = filename.split('-')
        batch, repetition, day = "", "", ""
        if len(splitted) > 0: batch = splitted[0]
        if len(splitted) > 1: repetition = splitted[1]
        if len(splitted) > 2: day = splitted[2]
        
        logger.info(f"Debug de Tamanhos - Aérea: {len(aerials)} | Raiz: {len(roots)}\n")
        
        max_len = max(len(aerials), len(roots))

        #Preenche com None (vazio) as listas que tiverem menos medidas
        aerials += [None] * (max_len - len(aerials))
        roots += [None] * (max_len - len(roots))
        
        df = pd.DataFrame({
        "Fold": fold,
        "Lote": batch,
        "Repeticao": repetition,
        "Dia": day,
        "Imagem": filename,
        "Plantula": range(1, max_len + 1),
        "MASK R-CNN - PA": aerials,
        "MASK R-CNN - RAIZ": roots,
        })

        dfs.append(df)

    if len(dfs) == 0:
        logger.warning("Nenhuma medida extraída! CSV não será gerado.")
        return
        
    df_final = pd.concat(dfs, ignore_index=True)
    # Ajustar caminho de saída f"C:/Users/RIAN/Desktop/Detectron2-measurements.csv"
    output_path = os.path.join(output_dir, 'measurements.csv')
    df_final.to_csv(output_path, index=False, sep=";")

# if __name__ == '__main__':
#     extract_measurements_from_masks(
#         cfg_path=r"train-maskr-cnn\results2\IS_cfg.pickle",
#         model_dir=r'train-maskr-cnn\results2',
#         output_dir=r"C:\Users\RIAN\Desktop",
#         images_path=r"train-maskr-cnn\dataset\test",
#         image_size=1280,
#         fold=1
#         )