import os
import cv2
import math
from ultralytics import YOLO
import numpy as np
import logging
from utils import bigger_skeleton, calculate_pixel_to_cm_ratio, dim_reduction
import pandas as pd
from natsort import natsorted

# Configuração básica
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logger = logging.getLogger(__name__)

def load_images(images_path: str) -> tuple[list,list]:
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
    model_path: str,
    output_dir: str,
    images_path:str,
    image_size: int,
    fold: int
    ):

    model = YOLO(model_path)
    images, filenames = load_images(images_path)

    results = model.predict(
        source=images,save=False, 
        conf=0.3, 
        iou=0.5, 
        show_boxes=False, 
        imgsz=image_size, 
        line_width=1, 
        retina_masks=True
    )
    dfs = []
    for result, filename in zip(results,filenames):
        # result.show()
        if result.masks is not None:
            cls = result.boxes.cls.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.data.cpu().numpy().astype(np.uint8)

            scale = calculate_pixel_to_cm_ratio(result.orig_img)
            if scale is None:
                logger.warning("Error: Fator de escala não encontrado!")
                continue

            print(result.orig_img.shape)
            print(result.orig_shape)
            print(f"Imagem: {filename}")
            print(f"Scale: {scale}")

            aerials_data = []
            roots_data = []
            for idx, class_id in enumerate(cls):
                mask = (masks[idx] > 0).astype(np.uint8) * 255
                x1, y1, x2, y2 = boxes[idx]
                cx = (x1 + x2) / 2.0  # Centro X
                cy = (y1 + y2) / 2.0  # Centro Y
                
                if class_id == 0: 
                    aerials_data.append({"mask": mask, "cx": cx, "cy": cy})
                else:
                    roots_data.append({"mask": mask, "cx": cx, "cy": cy})
            
            # Ordenar da esquerda para a direita (bom para manter a ordem visual no CSV)
            aerials_data.sort(key=lambda item: item["cx"])
            roots_data.sort(key=lambda item: item["cx"])

            matched_masks = []
            used_roots = set()

            # Pareamento guloso (Greedy Matching) pela distância dos centros
            for aerial in aerials_data:
                best_idx = -1
                best_dist = float('inf')
                
                for i, root in enumerate(roots_data):
                    if i in used_roots:
                        continue
                    
                    # Distância Euclidiana entre o centro da parte aérea e da raiz
                    dist = math.hypot(aerial["cx"] - root["cx"], aerial["cy"] - root["cy"])
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                
                # Se achou uma raiz para essa parte aérea, cria o par
                if best_idx != -1:
                    matched_masks.append((aerial["mask"], roots_data[best_idx]["mask"]))
                    used_roots.add(best_idx)
                else:
                    # Se não achou (YOLO falhou em detectar a raiz), pareia com máscara vazia (resultará em 0.0)
                    empty_mask = np.zeros_like(aerial["mask"])
                    matched_masks.append((aerial["mask"], empty_mask))

            # Adicionar raízes que sobraram sem parte aérea (YOLO falhou em detectar a folha)
            for i, root in enumerate(roots_data):
                if i not in used_roots:
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
        batch = ""
        repetition = ""
        day = ""
        if len(splitted) > 0: batch = splitted[0]
        if len(splitted) > 1: repetition = splitted[1]
        if len(splitted) > 2: day = splitted[2]
        
        logger.info(f"Debug de Tamanhos - Aérea: {len(aerials)} | Raiz: {len(roots)}\n")
        
        max_len = max(len(aerials), len(roots))

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
        "MASK R-CNN - RAIZ": roots
        })

        dfs.append(df)
        df_final = pd.concat(dfs, ignore_index=True)
        
    # Ajustar caminho de saída f"C:/Users/RIAN/Desktop/Detectron2-measurements.csv"
    output_path = os.path.join(output_dir, 'measurements.csv')
    df_final.to_csv(output_path, index=False, sep=";")
# extract_measurements_from_masks(model_path=r"C:\Users\RIAN\Desktop\TrainYOLO\last.pt",output_dir=r"C:\Users\RIAN\Desktop\TrainYOLO\output", images_path=r"dataset\L1\images", image_size=1024, fold=1)
