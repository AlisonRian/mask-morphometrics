import pickle
import os
import cv2
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

# Carrega a cfg utilizada no treinamento
cfg_saved_path = r"train-maskr-cnn\results2\IS_cfg.pickle"
with open(cfg_saved_path, 'rb') as f:
    cfg = pickle.load(f)

# Carrega o modelo treinado
output_dir = r'train-maskr-cnn\results2'
cfg.MODEL.WEIGHTS = os.path.join(output_dir, "model_final.pth")
# Define o limiar de confiança na inferência 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)
CLASSES = ['aerial', 'root']

# Caminho para as imagens de teste
image_path_1 = r"train-maskr-cnn\dataset\test\L5-R1-10D-F_jpg.rf.bd167be426251f4efd23495d9153af68.jpg"

img = cv2.imread(image_path_1)
outputs = predictor(img)
print('=====OUTPUTS=====')
print(outputs)
print('=================')

pred_classes = outputs['instances'].pred_classes.cpu().numpy()
print("=====CLASSES=====")
print(pred_classes)
print(type(pred_classes))
print('=================')

num_instances = len(pred_classes)
print(f"Total de objetos encontrados: {num_instances}")

if num_instances > 0:
    for i, class_id in enumerate(pred_classes):
        nome_classe = CLASSES[class_id]
        print(f"Instância {i+1}: ID {class_id} -> {nome_classe}")

    instances = outputs["instances"].to("cpu")
    if instances.has('pred_boxes'):
        instances.remove('pred_boxes')
    
    my_metadata = MetadataCatalog.get("my_dataset_test").set(thing_classes=CLASSES)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    v = Visualizer(img_rgb, metadata=my_metadata, scale=0.6)
    v = v.draw_instance_predictions(instances)
    img_bgr = cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR)

    cv2.namedWindow("v", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("v", 1920, 1080)
    cv2.imshow("v", img_bgr)
    cv2.waitKey(0)
else:
    print("Nenhum objeto detectado!")

cv2.destroyAllWindows()