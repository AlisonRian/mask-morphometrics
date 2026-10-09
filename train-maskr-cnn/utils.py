import cv2
import numpy as np
from cv2.typing import MatLike
import os
from collections import deque

def bfs_skeleton(skel: MatLike, start: tuple[int,int]) -> tuple[MatLike, dict[tuple[int,int], tuple[int,int]]]:
    h, w = skel.shape
    dist = -np.ones_like(skel, dtype=int)
    parent:dict[tuple[int,int], tuple[int,int]] = {}

    q = deque([start])
    dist[start] = 0

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while q:
        y, x = q.popleft()

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            if 0 <= ny < h and 0 <= nx < w:
                if skel[ny, nx] and dist[ny, nx] == -1:
                    dist[ny, nx] = dist[y, x] + 1
                    parent[(ny, nx)] = (y, x)
                    q.append((ny, nx))

    return dist, parent


def bigger_skeleton(binary: MatLike) -> list[tuple[int,int]]:
    ys, xs = np.where(binary) # type: ignore
    start = (ys[0], xs[0])

    if len(ys) == 0 or len(xs) == 0:
        return []
    
    dist1 = bfs_skeleton(binary, start)[0]  # type: ignore
    A = np.unravel_index(np.argmax(dist1), dist1.shape)

    dist2, parent = bfs_skeleton(binary, A)  # type: ignore
    B = np.unravel_index(np.argmax(dist2), dist2.shape)

    path:list[tuple[int,int]] = []
    current = B

    while current != A:
        path.append(current) # type: ignore
        current = parent[current] # type: ignore

    path.append(A) # type: ignore
    return path

def dim_reduction(image:MatLike, target_width:int = 1280) -> tuple[MatLike, float]:
    (height, width) = image.shape[:2]
    ratio =  target_width / width
    img_reduced = cv2.resize(image, (target_width, int(height * ratio)), interpolation=cv2.INTER_AREA)
    return img_reduced, 1.0/ratio

def preprocessing(image: MatLike) -> MatLike:
    # preprocessing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab = cv2.bilateralFilter(lab,d=7,sigmaColor=75,sigmaSpace=25)
    # enhancement
    lab = lab.astype(np.float32)
    lab[:,:,2] = cv2.normalize(lab[:,:,2] * 5, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
    lab[:,:,1] = cv2.normalize(lab[:,:,1] * 5, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
    lab = lab.astype(np.uint8)
    # edge detection
    edges = cv2.Canny(lab,threshold1=50,threshold2=150)
    # suavization
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.dilate(edges, kernel)
    return closed

def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    length = cv2.arcLength(contour, True)
    if length == 0:  
        return 0
    circularity = 4* np.pi * area / length**2
    return circularity

def calculate_pixel_to_cm_ratio(
    image: MatLike, 
    threshold: float = 0.8, 
    target_width: int = 1280,
    coin_diameter: float = 2.3
) -> float | None:
    # Reduz a imagem para facilitar o processamento
    img_reduced, inv_ratio = dim_reduction(image, target_width=target_width)
    # Detecta as bordas dos objetos e os contornos
    processed_image = preprocessing(img_reduced)
    contours = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Filtra os contornos por circularidade
    candidates = []
    for cnt in contours:
        c = calculate_circularity(cnt)
        if c >= threshold:
            candidates.append(cnt)
            
    if not candidates:
        print("Coin not found")
        return None

    # Pega o objeto circular de maior área
    coin = max(candidates, key=cv2.contourArea)
    (x_red, y_red), radius_reduced = cv2.minEnclosingCircle(coin)
    
    # Converte o raio e centro para a escala da imagem original
    radius_original = radius_reduced * inv_ratio

    diameter_pixels = 2 * radius_original
    scale = diameter_pixels / coin_diameter

    return scale

def compute_all_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Calcula IoU, Dice (F1), Precision e Recall em uma única passada.
    pred_mask e gt_mask: arrays binários (bool ou uint8 0/1) de mesma dimensão (H, W).
    Retorna: (iou, dice, precision, recall, TP, FP, FN)
    """
    pred_bin = pred_mask.astype(bool)
    gt_bin = gt_mask.astype(bool)

    pred_count = int(pred_bin.sum())
    gt_count = int(gt_bin.sum())

    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0, 1.0, 1.0, 0, 0, 0

    TP = int(np.logical_and(pred_bin, gt_bin).sum())
    FP = pred_count - TP
    FN = gt_count - TP

    total_area = pred_count + gt_count
    union = total_area - TP

    iou = float(TP / union) if union > 0 else 1.0
    dice = float((2.0 * TP) / total_area) if total_area > 0 else 1.0
    precision = float(TP / pred_count) if pred_count > 0 else 0.0
    recall = float(TP / gt_count) if gt_count > 0 else 0.0

    return iou, dice, precision, recall, TP, FP, FN

def compute_iou_and_dice(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Mantido para compatibilidade retroativa.
    Retorna: (iou, dice)
    """
    iou, dice, *_ = compute_all_metrics(pred_mask, gt_mask)
    return iou, dice

def calculate_precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Mantido para compatibilidade retroativa.
    Retorna: (precision, recall, TP, FP, FN)
    """
    _, _, precision, recall, TP, FP, FN = compute_all_metrics(pred_mask, gt_mask)
    return precision, recall, TP, FP, FN

def evaluate_semantic_metrics(predictor, coco, test_data_dir, dataset_name):
    from detectron2.data import MetadataCatalog
    
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    
    # Mapear os IDs da anotação COCO para os IDs internos (0-index) do Detectron2
    meta = MetadataCatalog.get(dataset_name)
    try:
        id_map = meta.thing_dataset_id_to_contiguous_id
    except AttributeError:
        id_map = {cat: i for i, cat in enumerate(sorted(cat_ids))}
    if id_map is None:
        id_map = {cat: i for i, cat in enumerate(sorted(cat_ids))}
        
    ious_per_class = {cat_id: [] for cat_id in cat_ids}
    dices_per_class = {cat_id: [] for cat_id in cat_ids}
    precisions_per_class = {cat_id: [] for cat_id in cat_ids}
    recalls_per_class = {cat_id: [] for cat_id in cat_ids}

    all_iou = []
    all_dice = []
    all_precision = []
    all_recall = []
        
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        
        img_path = os.path.join(test_data_dir, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        outputs = predictor(img)
        pred_instances = outputs["instances"].to("cpu")
        
        # --- A) Métricas por Classe ---
        for cat_id in cat_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
            anns = coco.loadAnns(ann_ids)
            
            gt_mask = np.zeros((height, width), dtype=bool)
            for ann in anns:
                m = coco.annToMask(ann)
                if len(m.shape) == 3:
                    m = np.any(m, axis=2)
                gt_mask = np.logical_or(gt_mask, m > 0)
                
            contiguous_id = id_map[cat_id]
            if len(pred_instances) > 0:
                masks = pred_instances.pred_masks.numpy()
                classes = pred_instances.pred_classes.numpy()
                class_masks = masks[classes == contiguous_id]
                if len(class_masks) > 0:
                    pred_bin = np.any(class_masks, axis=0).astype(bool)
                else:
                    pred_bin = np.zeros((height, width), dtype=bool)
            else:
                pred_bin = np.zeros((height, width), dtype=bool)
                
            if pred_bin.shape != gt_mask.shape:
                pred_bin = cv2.resize(pred_bin.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                
            iou, dice, prec, rec, *_ = compute_all_metrics(pred_bin, gt_mask)

            ious_per_class[cat_id].append(iou)
            dices_per_class[cat_id].append(dice)
            precisions_per_class[cat_id].append(prec)
            recalls_per_class[cat_id].append(rec)

        # --- B) Métrica Global (instâncias unificadas) ---
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        gt_mask = np.zeros((height, width), dtype=bool)
        for ann in anns:
            m = coco.annToMask(ann)
            if len(m.shape) == 3:
                m = np.any(m, axis=2)
            gt_mask = np.logical_or(gt_mask, m > 0)
            
        if len(pred_instances) > 0:
            pred_masks = pred_instances.pred_masks.numpy()
            pred_bin = np.any(pred_masks, axis=0).astype(bool)
        else:
            pred_bin = np.zeros((height, width), dtype=bool)
            
        if pred_bin.shape != gt_mask.shape:
            pred_bin = cv2.resize(pred_bin.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            
        iou, dice, prec, rec, *_ = compute_all_metrics(pred_bin, gt_mask)

        all_iou.append(iou)
        all_dice.append(dice)
        all_precision.append(prec)
        all_recall.append(rec)
            
    # Resultado Final
    print("\n" + "="*60)
    print("MÉTRICAS SEMÂNTICAS (mIoU, mDice, Precision, Recall)")
    miou_per_class = {}
    mdice_per_class = {}
    mprec_per_class = {}
    mrec_per_class = {}

    for cat_id in cat_ids:
        cat_name = coco.loadCats(cat_id)[0]['name']
        mean_iou = float(np.mean(ious_per_class[cat_id])) if len(ious_per_class[cat_id]) > 0 else 0.0
        mean_dice = float(np.mean(dices_per_class[cat_id])) if len(dices_per_class[cat_id]) > 0 else 0.0
        mean_prec = float(np.mean(precisions_per_class[cat_id])) if len(precisions_per_class[cat_id]) > 0 else 0.0
        mean_rec = float(np.mean(recalls_per_class[cat_id])) if len(recalls_per_class[cat_id]) > 0 else 0.0

        miou_per_class[cat_name] = mean_iou
        mdice_per_class[cat_name] = mean_dice
        mprec_per_class[cat_name] = mean_prec
        mrec_per_class[cat_name] = mean_rec

        print(f"Classe [{cat_name}]: IoU = {mean_iou:.4f}, Dice = {mean_dice:.4f}, Precision = {mean_prec:.4f}, Recall = {mean_rec:.4f}")

    global_miou = float(np.mean(all_iou)) if len(all_iou) > 0 else 0.0
    global_mdice = float(np.mean(all_dice)) if len(all_dice) > 0 else 0.0
    global_mprec = float(np.mean(all_precision)) if len(all_precision) > 0 else 0.0
    global_mrec = float(np.mean(all_recall)) if len(all_recall) > 0 else 0.0

    print("-" * 60)
    print(f"Global: IoU = {global_miou:.4f}, Dice = {global_mdice:.4f}, Precision = {global_mprec:.4f}, Recall = {global_mrec:.4f}")
    print("="*60 + "\n")

    return {
        'mIoU': global_miou,
        'mDice': global_mdice,
        'Precision': global_mprec,
        'Recall': global_mrec,
        'mIoU_per_class': miou_per_class,
        'mDice_per_class': mdice_per_class,
        'Precision_per_class': mprec_per_class,
        'Recall_per_class': mrec_per_class
    }

