import torch
import torch.nn as nn
from RTDETRv2.rtdetrv2_pytorch.src.core import YAMLConfig
import argparse
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def get_rtdetrv2_model(rtdetrv2_path, config_path, device, resume = None):
    """
    Loads the RT-DETR V2 model from a checkpoint and prepares it for deployment.
    """
    # Load checkpoint with the appropriate map location
    checkpoint = torch.load(rtdetrv2_path, map_location=device)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    # Load the state dictionary into the model as defined in the configuration
    cfg = YAMLConfig(config_path, resume=resume)
    cfg.model.load_state_dict(state)

    # Define a wrapper model that deploys both the backbone and its postprocessor
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    return model

class RTDETRv2:
    def __init__(self, rtdetrv2_path, config_path, device='cuda', resume=None):
        """
        Initializes the RT-DETR V2 model.
        """
        self.device = device
        self.model = get_rtdetrv2_model(rtdetrv2_path, config_path, device, resume)
        self.model.eval()
        self.transform = T.Compose([T.Resize((640, 640)),  # Resize image to network's expected input size.
                            T.ToTensor(),          # Convert image to tensor.
                            ])

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
        # check if image is a path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image or a path to an image file.")
        w,h = image.size
        orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(self.device)
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(img_tensor, orig_size)
        
        labels, boxes, scores = predictions
        labels = labels.cpu().numpy()
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        # Filter predictions by confidence threshold if specified
        if conf_thr is not None:
            mask = scores >= conf_thr
            labels = labels[mask]
            boxes = boxes[mask]
            scores = scores[mask]

        out_dict = {
            "labels": labels,
            "boxes": boxes,
            "scores": scores
        }

        # Add normalized boxes if requested
        if return_norm_boxes:
            norm_boxes = boxes / np.array([w, h, w, h], dtype=np.float32)
            out_dict["norm_boxes"] = norm_boxes

        return out_dict
    
def main():
    """
    Main function to test RT-DETRv2 inference and visualization.
    """
    parser = argparse.ArgumentParser(description="RT-DETRv2 Inference Test")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda or cpu)')
    args = parser.parse_args()

    try:
        # Initialize the model
        detector = RTDETRv2(
            rtdetrv2_path=args.checkpoint,
            config_path=args.config,
            device=args.device
        )
        
        # Perform inference
        predictions = detector.infer(args.image)
        
        # Print predictions
        print("Inference Results:")
        for label, box, score in zip(predictions["labels"], predictions["boxes"], predictions["scores"]):
            print("Label:", label, "Box:", box, "Score:", score)
        
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return

if __name__ == "__main__":
    main()