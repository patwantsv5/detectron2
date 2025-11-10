import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

# Path to your trained weights
weights_path = "C:\\Users\\user\\Desktop\\detectron2\\detectron2\\model_final.pth"
# Folder of images
image_folder = "C:\\Users\\user\\Desktop\\detectron2\\detectron2\\shipyard\\test"

cfg = get_cfg()
cfg.merge_from_file("C:\\Users\\user\\Desktop\\detectron2\\detectron2\\rotated_bbox_config.yaml")
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE = "cpu"  # or "cuda" if GPU is available

predictor = DefaultPredictor(cfg)

for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1])
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        save_path = os.path.join(image_folder, "pred_" + filename)
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        print(f"Saved: {save_path}")
