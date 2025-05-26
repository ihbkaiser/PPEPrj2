# Dual-Model Fusion for PPE Detection

[![Build Status](https://img.shields.io/github/actions/workflow/status/ihbkaiser/PPEPrj2/ci.yml?branch=main&style=flat-square)](https://github.com/ihbkaiser/PPEPrj2/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-orange.svg?style=flat-square)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red.svg?style=flat-square)](https://streamlit.io/)

**Dual-Model Fusion for Personal Protective Equipment (PPE) Detection** is a cutting-edge computer vision project that leverages the power of two state-of-the-art object detection models, **Hyper-YOLOT** and **RT-DETRv2**, to accurately identify PPE items (e.g., hardhats, vests, gloves) in images. By combining predictions using Weighted Boxes Fusion (WBF), this project achieves robust and reliable detection, making it ideal for safety compliance monitoring in industrial settings.

This repository is part of **Project 2 - 20242** and provides a user-friendly Streamlit web interface for real-time PPE detection, complete with example images and customizable inference parameters.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- **Dual-Model Architecture**: Combines **Hyper-YOLOT** (improved on YOLO) and **RT-DETRv2** for enhanced detection accuracy.
- **Weighted Boxes Fusion (WBF)**: Fuses predictions from both models to reduce false positives and improve precision.
- **Interactive Web Interface**: Built with Streamlit, allowing users to upload images, select examples, and adjust inference parameters (confidence threshold, model weights).
- **Custom PPE Labels**: Supports detection of PPE items using a custom `coco_classes` dictionary (e.g., `boots`, `gloves`, `hardhat`, `vest`).
- **Pre-trained Weights**: Includes ready-to-use weights for **Hyper-YOLOT** and **RT-DETRv2** (with `rtdetrv2_r101vd_6x_coco` config).

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ihbkaiser/PPEPrj2.git
   cd PPEPrj2
   ```

2. **Set Up a Conda Environment**:
   ```bash
   conda create -n ppe python=3.8
   conda activate ppe
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
   pip install streamlit pillow numpy ensemble-boxes
   ```
   *Note*: Adjust the PyTorch version based on your CUDA setup (e.g., `torch==2.5.0+cu118` for CUDA 11.8). Check [PyTorch’s website](https://pytorch.org/get-started/locally/) for details.


4. **Verify Model Files**:
   Ensure the following files are in the correct paths:
   - `ckpt/hyper-yolo.pt`
   - `ckpt/rtdetrv2.pth`
   - `RTDETRv2/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml`  

   Download pre-trained weights from the provided links (see [Training](#training)).

## Dataset
The project uses the **PPE Dataset** from Roboflow:
- **Source**: [PPE Dataset on Roboflow](https://universe.roboflow.com/ppe-yngjj/ppe-vum8g/dataset/10)
- **Classes**: Includes PPE items like `boots`, `gloves`, `hardhat`, `vest`, and negative classes (`no_boots`, `no_gloves`, `no_hardhat`, `no_vest`), plus `person`.

**Steps**:
1. Download the dataset from the Roboflow link.
2. Extract it to a directory (e.g., `data/ppe_dataset`).
3. Update the dataset configuration files for training (see [Training](#training)).

## Training
To train the models, follow these steps:

### Hyper-YOLOT
1. **Update Data Configuration**:
   - Edit `hyperyolot/data.yaml` to point to your dataset path:
     ```yaml
     train: /path/to/ppe_dataset/train/images
     val: /path/to/ppe_dataset/val/images
     nc: 9  # Number of classes (boots, gloves, etc.)
     names: ['boots', 'gloves', 'hardhat', 'no_boots', 'no_gloves', 'no_hardhat', 'no_vest', 'person', 'vest']
     ```
   - Ensure the path matches your extracted dataset.

2. **Train the Model**:
   Follow the YOLO training format (refer to `ultralytics` documentation):
   ```bash
   python3 hyperyolot/ultralytics/models/yolo/detect/train.py
   ```

### RT-DETRv2
**Refer to Official Repository**:
   - Follow the training instructions in the [RT-DETRv2 GitHub repository](https://github.com/supervisly-ecosystem/RTDETRv2).
   - Update the configuration file (`rtdetrv2_r101vd_6x_coco.yml`) to point to your dataset.

### Pre-trained Weights
We provide pre-trained weights for convenience:
- **Hyper-YOLOT**: `ckpt/hyper-yolo.pt`
- **RT-DETRv2**: `ckpt/rtdetrv2.pth` (with config `rtdetrv2_r101vd_6x_coco.yml`)
- Download links: [LINK]()

Place these files in the `ckpt/` directory.

## Usage
1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```

2. **Access the App**:
Open `http://localhost:8501` in your local browser.

3. **Interact with the App**:
   - **Select Image**: Upload an image or click a thumbnail from the example images (in `examples/`).
   - **Set Parameters**: Adjust confidence threshold and model weights in the sidebar.
   - **Run Inference**: Click the "Inference" button to process the image.
   - **View Results**: See fused predictions with bounding boxes labeled as PPE items (e.g., `hardhat`, `vest`). Optionally view raw model outputs.

## Contributing
We welcome contributions to enhance the project! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make changes and commit (`git commit -m "Add your feature"`).
4. Push to your fork (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description.

Please follow our [Code of Conduct](#) and report issues on the [GitHub Issues page](https://github.com/ihbkaiser/PPEPrj2/issues).

## License
This project is licensed under the MIT License
