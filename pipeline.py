import torch
from RTDETRv2.rtdetrv2 import RTDETRv2
from hyperyolot.hyper_yolot import HyperYOLOT
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import os
import argparse
import torch.serialization

# Allowlist Ultralytics global for YOLO model loading
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

# Project-specific paths
BASE_DIR = "/home/ducanh/Credit/MultiModelPPE/PPE-Pipeline-Project2"
YOLO_PATH = os.path.join(BASE_DIR, "ckpt", "hyper-yolo.pt")
RTDETR_PATH = os.path.join(BASE_DIR, "ckpt", "rtdetrv2.pth")
RTDETR_CONFIG = os.path.join(BASE_DIR, "RTDETRv2", "rtdetrv2_pytorch", "configs", "rtdetrv2", "rtdetrv2_r101vd_6x_coco.yml")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Label mapping for PPE detection (update based on your models)
LABEL_MAP = {
    "yolo_to_rtdetr": {
        0: 0,  # helmet -> helmet
        1: 1,  # vest -> vest
        2: 2,  # gloves -> gloves
        # Add more mappings as needed
    }
}

# PPE class names (update based on your dataset)
CLASS_NAMES = ["Helmet", "Vest", "Gloves"]  # Replace with actual classes

# Visualization Function
def visualize_image(image, predictions, title="Detected Objects"):
    """
    Draws bounding boxes, labels, and scores on the image.
    Args:
        image: PIL Image.
        predictions: Dictionary with labels, boxes, and scores.
        title: Title for the image (used in filename).
    Returns:
        PIL Image with drawn boxes.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
        x_min, y_min, x_max, y_max = box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline="red",
            width=3
        )
        label_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class {int(label)}"
        text = f"{label_name}: {score:.2f}"
        draw.text(
            (x_min, y_min - 25),
            text,
            fill="red",
            font=font
        )
    
    return img

# Label Mapping Function
def map_labels(labels, mapping):
    """
    Maps labels from one model to another’s label set.
    Args:
        labels: NumPy array of labels.
        mapping: Dictionary mapping source labels to target labels.
    Returns:
        Mapped labels as NumPy array.
    """
    mapped_labels = np.array([mapping.get(int(label), int(label)) for label in labels], dtype=np.int32)
    return mapped_labels

# Model Loading Function
def load_models(yolo_path, rtdetr_path, rtdetr_config):
    """
    Loads both models.
    Args:
        yolo_path: Path to YOLO model file.
        rtdetr_path: Path to RTDETRv2 checkpoint.
        rtdetr_config: Path to RTDETRv2 config YAML.
    Returns:
        Tuple of (yolo_model, rtdetr_model).
    """
    if not all(os.path.exists(p) for p in [yolo_path, rtdetr_path, rtdetr_config]):
        print(f"Error: One or more model files not found: {yolo_path}, {rtdetr_path}, {rtdetr_config}")
        return None, None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        yolo_model = HyperYOLOT(yolo_path, device=device)
        print("Loaded HyperYOLOT model successfully")
    except Exception as e:
        print(f"Failed to load HyperYOLOT model: {e}")
        return None, None
    
    try:
        rtdetr_model = RTDETRv2(rtdetr_path, rtdetr_config, device=device)
        print("Loaded RTDETRv2 model successfully")
    except Exception as e:
        print(f"Failed to load RTDETRv2 model: {e}")
        return None, None
    
    return yolo_model, rtdetr_model

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PPE Detection with HyperYOLOT and RTDETRv2 using WBF")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf-thr", type=float, default=0.5, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--yolo-weight", type=float, default=1.0, help="Weight for HyperYOLOT in WBF (0.0 to 2.0)")
    parser.add_argument("--rtdetr-weight", type=float, default=1.0, help="Weight for RTDETRv2 in WBF (0.0 to 2.0)")
    parser.add_argument("--show-raw", action="store_true", help="Save raw model outputs")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not (0.0 <= args.conf_thr <= 1.0):
        print("Error: Confidence threshold must be between 0.0 and 1.0")
        return
    
    if not (0.0 <= args.yolo_weight <= 2.0 and 0.0 <= args.rtdetr_weight <= 2.0):
        print("Error: Model weights must be between 0.0 and 2.0")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load image
        image = Image.open(args.image).convert('RGB')
        print(f"Loaded image: {args.image} (size: {image.size})")

        # Load models
        yolo_model, rtdetr_model = load_models(YOLO_PATH, RTDETR_PATH, RTDETR_CONFIG)
        if yolo_model is None or rtdetr_model is None:
            return

        # Run inference
        print("Running inference...")
        yolo_preds = yolo_model.infer(image, return_norm_boxes=True, conf_thr=args.conf_thr)
        rtdetr_preds = rtdetr_model.infer(image, return_norm_boxes=True, conf_thr=args.conf_thr)
        print("Inference completed")

        # Map YOLO labels to RTDETRv2 labels
        yolo_preds["labels"] = map_labels(yolo_preds["labels"], LABEL_MAP["yolo_to_rtdetr"])

        # Weighted Boxes Fusion
        w, h = image.size
        boxes_list = [
            yolo_preds["norm_boxes"],
            rtdetr_preds["norm_boxes"]
        ]
        scores_list = [
            yolo_preds["scores"],
            rtdetr_preds["scores"]
        ]
        labels_list = [
            yolo_preds["labels"],
            rtdetr_preds["labels"]
        ]
        weights = [args.yolo_weight, args.rtdetr_weight]

        print("Performing Weighted Boxes Fusion...")
        try:
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights,
                iou_thr=0.5,
                skip_box_thr=0.0001
            )
        except Exception as e:
            print(f"WBF failed: {e}")
            return
        print("WBF completed")

        # Denormalize fused boxes
        fused_boxes = fused_boxes * np.array([w, h, w, h], dtype=np.float32)
        fused_preds = {
            "labels": fused_labels.astype(int),
            "boxes": fused_boxes,
            "scores": fused_scores
        }

        # Visualize and save fused results
        fused_image = visualize_image(image, fused_preds, title="Fused Predictions")
        fused_output_path = os.path.join(OUTPUT_DIR, "fused_predictions.png")
        fused_image.save(fused_output_path)
        print(f"Saved fused predictions: {fused_output_path}")

        # Print fused prediction details
        print("\nFused Prediction Details:")
        for label, box, score in zip(fused_preds["labels"], fused_preds["boxes"], fused_preds["scores"]):
            label_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class {int(label)}"
            print(f"Label: {label_name}, Box: {box.tolist()}, Score: {score:.4f}")

        # Save and print raw predictions if requested
        if args.show_raw:
            yolo_image = visualize_image(image, yolo_preds, title="HyperYOLOT Predictions")
            yolo_output_path = os.path.join(OUTPUT_DIR, "yolo_predictions.png")
            yolo_image.save(yolo_output_path)
            print(f"Saved HyperYOLOT predictions: {yolo_output_path}")

            rtdetr_image = visualize_image(image, rtdetr_preds, title="RTDETRv2 Predictions")
            rtdetr_output_path = os.path.join(OUTPUT_DIR, "rtdetr_predictions.png")
            rtdetr_image.save(rtdetr_output_path)
            print(f"Saved RTDETRv2 predictions: {rtdetr_output_path}")

            print("\nHyperYOLOT Prediction Details:")
            for label, box, score in zip(yolo_preds["labels"], yolo_preds["boxes"], yolo_preds["scores"]):
                label_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class {int(label)}"
                print(f"Label: {label_name}, Box: {box.tolist()}, Score: {score:.4f}")

            print("\nRTDETRv2 Prediction Details:")
            for label, box, score in zip(rtdetr_preds["labels"], rtdetr_preds["boxes"], rtdetr_preds["scores"]):
                label_name = CLASS_NAMES[int(label)] if int(label) < len(CLASS_NAMES) else f"Class {int(label)}"
                print(f"Label: {label_name}, Box: {box.tolist()}, Score: {score:.4f}")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()