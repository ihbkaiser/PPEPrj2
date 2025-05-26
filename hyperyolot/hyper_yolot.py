import os
import glob
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO
from PIL import Image
import torch
class HyperYOLOT:
    def __init__(self, model_path, device='cuda'):
        """
        Initializes the HyperYOLO-T model.
        Args:
            model_path: Path to the YOLO model file.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.device = device
        self.model = YOLO(model_path).to(device)


    def infer(self, image, return_norm_boxes=False, conf_thr=None):
        """
        Performs inference on a single image.
        Args:
            image: A PIL Image or a path to an image file.
            return_norm_boxes: If True, includes normalized bounding boxes in the output.
            conf_thr: Confidence threshold to filter predictions (e.g., 0.5). If None, no filtering is applied.
        Returns:
            A dictionary with:
                - labels: Array of class labels (int).
                - boxes: Array of bounding boxes in [x_min, y_min, x_max, y_max] format (float).
                - scores: Array of confidence scores (float, 0 to 1).
                - norm_boxes (optional): Array of normalized bounding boxes if return_norm_boxes is True.
        Raises:
            ValueError: If the input is invalid or image loading fails.
        """
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to open image file: {e}")
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image or a path to an image file.")

        w, h = image.size
        results = self.model(image, device=self.device)
        result = results[0]
        boxes_obj = result.boxes
        xyxy = boxes_obj.xyxy.cpu().numpy() if hasattr(boxes_obj.xyxy, 'cpu') else boxes_obj.xyxy
        scores = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj.conf, 'cpu') else boxes_obj.conf
        labels = boxes_obj.cls.cpu().numpy() if hasattr(boxes_obj.cls, 'cpu') else boxes_obj.cls

        # Filter predictions by confidence threshold if specified
        if conf_thr is not None:
            mask = scores >= conf_thr
            labels = labels[mask]
            xyxy = xyxy[mask]
            scores = scores[mask]

        out_dict = {
            "labels": labels,
            "boxes": xyxy,
            "scores": scores
        }

        # Add normalized boxes if requested
        if return_norm_boxes:
            norm_boxes = xyxy / np.array([w, h, w, h], dtype=np.float32)
            out_dict["norm_boxes"] = norm_boxes

        return out_dict
    
if __name__ == "__main__":
    model_path = "ckpt/hyper-yolo.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = HyperYOLOT(model_path, device=device)
    print(f"Model loaded successfully on {device}")