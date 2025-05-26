import streamlit as st
import torch
from RTDETRv2.rtdetrv2 import RTDETRv2
from hyperyolot.hyper_yolot import HyperYOLOT
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(".", "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting Streamlit app")


# Project-specific paths
BASE_DIR = "."
YOLO_PATH = os.path.join(BASE_DIR, "ckpt", "hyper-yolo.pt")
RTDETR_PATH = os.path.join(BASE_DIR, "ckpt", "rtdetrv2.pth")
RTDETR_CONFIG = os.path.join(BASE_DIR, "RTDETRv2", "rtdetrv2_pytorch", "configs", "rtdetrv2", "rtdetrv2_r101vd_6x_coco.yml")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")

# COCO classes dictionary
coco_classes = {
    0: 'boots', 1: 'gloves', 2: 'hardhat', 3: 'no_boots', 4: 'no_gloves',
    5: 'no_hardhat', 6: 'no_vest', 7: 'person', 8: 'vest'
}

# Visualization Function
def visualize_image(image, predictions, title="Detected Objects"):
    """
    Draws bounding boxes, labels, and scores on the image.
    Args:
        image: PIL Image.
        predictions: Dictionary with labels, boxes, and scores.
        title: Title for the image (used for caption).
    Returns:
        PIL Image with drawn boxes.
    """
    logger.debug(f"Visualizing image with {len(predictions['boxes'])} predictions")
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        logger.debug("Loaded arial.ttf font")
    except:
        font = ImageFont.load_default()
        logger.warning("Failed to load arial.ttf, using default font")

    for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
        x_min, y_min, x_max, y_max = box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline="red",
            width=3
        )
        label_name = coco_classes.get(int(label), f"Class {int(label)}")
        text = f"{label_name}: {score:.2f}"
        draw.text(
            (x_min, y_min - 25),
            text,
            fill="red",
            font=font
        )
    
    logger.debug("Image visualization completed")
    return img

# Model Loading Function
@st.cache_resource
def load_models(yolo_path, rtdetr_path, rtdetr_config):
    """
    Loads both models.
    Args:
        yolo_path: Path to the YOLO model file.
        rtdetr_path: Path to RTDETRv2 checkpoint.
        rtdetr_config: Path to RTDETRv2 config YAML.
    Returns:
        Tuple of (yolo_model, rtdetr_model).
    """
    logger.debug("Loading model files")
    if not all(os.path.exists(p) for p in [yolo_path, rtdetr_path, rtdetr_config]):
        logger.error(f"One or more model files not found: {yolo_path}, {rtdetr_path}, {rtdetr_config}")
        st.error("One or more model files not found. Please check the paths.")
        return None, None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    try:
        yolo_model = HyperYOLOT(yolo_path, device=device)
        logger.info("Loaded HyperYOLOT model successfully")
    except Exception as e:
        logger.error(f"Failed to load HyperYOLOT model: {e}")
        st.error(f"Failed to load HyperYOLOT model: {e}")
        return None, None

    try:
        rtdetr_model = RTDETRv2(rtdetr_path, rtdetr_config, device=device)
        logger.info("Loaded RTDETRv2 model successfully")
    except Exception as e:
        logger.error(f"Failed to load RTDETRv2 model: {e}")
        st.error(f"Failed to load RTDETRv2 model: {e}")
        return None, None
    
    return yolo_model, rtdetr_model

def main():
    logger.info("Entering main function")
    st.title("PPE Detection with HyperYOLOT and RTDETRv2")
    st.write("Upload an image or click an example below to detect PPE items (e.g., hardhat, vest, gloves) using two models with Weighted Boxes Fusion (WBF). Press 'Inference' to process the selected image.")

    # Sidebar for parameters
    st.sidebar.header("Inference Parameters")
    conf_thr = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    yolo_weight = st.sidebar.slider("HyperYOLOT Weight", 0.0, 2.0, 1.0, 0.1)
    rtdetr_weight = st.sidebar.slider("RTDETRv2 Weight", 0.0, 2.0, 1.0, 0.1)
    show_raw = st.sidebar.checkbox("Show Raw Model Outputs", value=False)
    logger.debug(f"Parameters set: conf_thr={conf_thr}, yolo_weight={yolo_weight}, rtdetr_weight={rtdetr_weight}, show_raw={show_raw}")

    # Example images as thumbnails
    st.subheader("Example Images")
    example_images = []
    if os.path.exists(EXAMPLES_DIR):
        example_images = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
        logger.debug(f"Found {len(example_images)} example images in {EXAMPLES_DIR}")
    else:
        logger.warning(f"Examples directory not found: {EXAMPLES_DIR}")
        st.warning("No example images found in examples/ directory.")

    selected_image_path = None
    if example_images:
        cols = st.columns(4)  # 4 thumbnails per row
        for i, img_name in enumerate(example_images):
            img_path = os.path.join(EXAMPLES_DIR, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img.thumbnail((100, 100))  # Thumbnail size
                with cols[i % 4]:
                    if st.image(img, caption=img_name, use_column_width=True):
                        if st.button(f"Select {img_name}", key=f"example_{i}"):
                            selected_image_path = img_path
                            logger.info(f"Selected example image: {img_path}")
            except Exception as e:
                logger.error(f"Failed to load example image {img_path}: {e}")
                st.warning(f"Failed to load {img_name}")
    else:
        logger.debug("No example images to display")

    # Image selection
    image_source = st.radio("Select Image Source", ["Upload Image", "Use Example Image"])
    logger.debug(f"Image source selected: {image_source}")

    image = None
    if image_source == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            logger.info("Loaded uploaded image")
    elif selected_image_path:
        try:
            image = Image.open(selected_image_path).convert('RGB')
            logger.info(f"Loaded example image: {selected_image_path}")
        except Exception as e:
            logger.error(f"Failed to load selected example image {selected_image_path}: {e}")
            st.error(f"Failed to load selected example image: {e}")

    if image is not None:
        st.image(image, caption="Selected Image", use_column_width=True)

        # Inference button
        if st.button("Inference"):
            logger.info("Inference button clicked")
            # Load models
            logger.debug("Loading models")
            yolo_model, rtdetr_model = load_models(YOLO_PATH, RTDETR_PATH, RTDETR_CONFIG)
            if yolo_model is None or rtdetr_model is None:
                logger.error("Model loading failed, exiting")
                return

            try:
                # Run inference
                logger.info("Starting inference")
                with st.spinner("Running inference..."):
                    yolo_preds = yolo_model.infer(image, return_norm_boxes=True, conf_thr=conf_thr)
                    rtdetr_preds = rtdetr_model.infer(image, return_norm_boxes=True, conf_thr=conf_thr)
                logger.info("Inference completed")

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
                weights = [yolo_weight, rtdetr_weight]
                logger.debug(f"WBF inputs: {len(boxes_list[0])} YOLO boxes, {len(boxes_list[1])} RTDETR boxes")

                try:
                    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                        boxes_list,
                        scores_list,
                        labels_list,
                        weights=weights,
                        iou_thr=0.5,
                        skip_box_thr=0.0001
                    )
                    logger.info("WBF completed successfully")
                except Exception as e:
                    logger.error(f"WBF failed: {e}")
                    st.error(f"WBF failed: {e}")
                    return

                # Denormalize fused boxes
                fused_boxes = fused_boxes * np.array([w, h, w, h], dtype=np.float32)
                fused_preds = {
                    "labels": fused_labels.astype(int),
                    "boxes": fused_boxes,
                    "scores": fused_scores
                }
                logger.debug(f"Fused predictions: {len(fused_boxes)} boxes")

                # Display fused results
                st.subheader("Fused Predictions (WBF)")
                fused_image = visualize_image(image, fused_preds, title="Fused Predictions")
                st.image(fused_image, caption="Fused Predictions (WBF)", use_column_width=True)
                logger.info("Displayed fused predictions")

                # Show raw predictions if requested
                if show_raw:
                    logger.debug("Generating raw model outputs")
                    st.subheader("Raw Model Outputs")
                    col1, col2 = st.columns(2)
                    with col1:
                        yolo_image = visualize_image(image, yolo_preds, title="HyperYOLOT Predictions")
                        st.image(yolo_image, caption="HyperYOLOT Predictions", use_column_width=True)
                        logger.debug("Displayed HyperYOLOT raw predictions")
                    with col2:
                        rtdetr_image = visualize_image(image, rtdetr_preds, title="RTDETRv2 Predictions")
                        st.image(rtdetr_image, caption="RTDETRv2 Predictions", use_column_width=True)
                        logger.debug("Displayed RTDETRv2 raw predictions")

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    logger.info("Executing main script")
    main()