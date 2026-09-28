import detectron2
from detectron2.projects import point_rend

import cv2
import numpy as np
import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

cfg = get_cfg()

# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml")
cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.WEIGHTS = os.path.join(r'C:\Users\RIAN\detectron2\train-maskr-cnn\results_point_rend', "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
predictor = DefaultPredictor(cfg)

img = cv2.imread(r"C:\Users\RIAN\Desktop\seedling_images\valid\L2R1-7.jpg")
h, w, _ = img.shape
outputs = predictor(img)
print(outputs)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
v = Visualizer(img_rgb,
                   metadata={}, 
                   scale=1.0, 
                   instance_mode=ColorMode.IMAGE
    )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img_bgr = cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR)
cv2.imwrite("C:/Users/RIAN/Desktop/result_point_rend.png",cv2.cvtColor(out.get_image(), cv2.COLOR_BGR2RGB))
cv2.imshow("v",img_bgr)
cv2.waitKey(0)