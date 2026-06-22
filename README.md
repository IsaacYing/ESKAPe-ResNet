# ESKAPe-ResNet
Rapid, label-free identification of ESKAPE pathogens from bright-field microscopy at single-cell resolution.

## Overview

Rapid and accurate pathogen identification is crucial for the clinical management of infectious diseases, particularly sepsis and severe respiratory infections, yet standard clinical workflows remain slow and resource-intensive. Here, we developed an automated, high-throughput imaging platform built on standard, clinically accessible bright-field microscopy, and generated a large dataset comprising 24.9 million label-free bacterial cells across six focal pathogens. Leveraging this resource, we trained a neural network (ESKAPe-ResNet) to identify ESKAPe species at the single-bacterium level.

This work establishes the proof-of-principle for label-free, hardware-minimal rapid pathogen identification, providing a clinically deployable workflow to expedite diagnosis and reduce mortality in severe bacterial infections.

## Model

ESKAPe-ResNet identifies six ESKAPe pathogens from single bacterial cells(e for *Escherichia coli* substituting *Enterobacter spp.*):

| Class | Species |
|:-----:|:--------|
| *Efm* | *Enterococcus faecium* |
| *Sau* | *Staphylococcus aureus* |
| *Kpn* | *Klebsiella pneumoniae* |
| *Aba* | *Acinetobacter baumannii* |
| *Pae* | *Pseudomonas aeruginosa* |
| *Eco* | *Escherichia coli* |

## Repository Structure
```
ESKAPe-ResNet/
├── ESKAPe_Resnet.pth          # Pre-trained model weights (PyTorch)
├── ESKAPe_Resnet.py           # Model training script
├── build_training_batches.py  # Dataset batch construction for model training
├── NT_controller.py           # Microscope control software
└── README.md                   # This file
```

## Imaging Platform

### NikonTi-Brightfield-AutoImaging

The automated bright-field microscopy platform is built on a Nikon Ti2 inverted microscope equipped with motorized stage, LED illumination, and high-resolution CMOS camera. Image acquisition is controlled by a custom Python-based automation framework (`NT_controller.py`) that integrates hardware control, automated stage movement, plate scanning, and autofocus into a unified acquisition environment.

**Key capabilities:**
- **Automated bright-field imaging** — single-shot acquisition with adjustable exposure and ImageJ-compatible TIFF export
- **Autofocus** — two modes: (i) frequency-based focus estimation via Fourier transformation of Z-stacks; (ii) AI-assisted focal prediction from bright-field images
- **Multi-well plate scanning** — supports 6/12/24/48/96-well plates with four-point calibration for coordinate generation
- **Metadata-rich output** — each TIFF includes objective info, pixel size, numerical aperture, exposure, XY coordinates, Z positions, and timestamps

**Typical acquisition workflow:**
1. Configure microscope parameters and plate type
2. Calibrate plate coordinates (four-point)
3. Select target wells and configure autofocus mode
4. Start automated acquisition — the software performs autofocus, imaging, and data saving unattended

This platform was used to generate the 24.9 million single-cell image dataset for ESKAPe-ResNet training and validation.

## Software Environment and Dependencies

All computational analyses were performed in **Python 3.8.20** with **PyTorch 1.13.1** (CUDA 11.6, cuDNN 8.4.0) using NVIDIA A100 (40 GB) GPUs with CUDA acceleration.

Key dependencies:

| Package | Version |
|:--------|:--------|
| PyTorch | 1.13.1 |
| torchvision | 0.14.1 |
| NumPy | 1.24.4 |
| scikit-learn | 1.3.2 |
| Pillow | 10.4.0 |
| matplotlib | 3.7.5 |
| seaborn | 0.13.2 |
| Omnipose | 1.0.6 |

Install dependencies:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install numpy==1.24.4 scikit-learn==1.3.2 Pillow==10.4.0 matplotlib==3.7.5 seaborn==0.13.2
pip install omnipose==1.0.6
```
> **Note:** Adjust the PyTorch CUDA version (`cu116`) according to your local CUDA installation.

## Usage

### Microscope Image Acquisition

For automated image acquisition using the bright-field microscopy platform:

```bash
python NT_controller.py
```

> **Note:** Ensure your microscope hardware is properly connected and configured. Adjust device parameters (e.g., COM port, camera settings) in `NT_controller.py` as needed.

### Dataset Preparation

Before training, run the dataset batch construction script:

```bash
python build_training_batches.py
```

> **Note:** Modify the input image paths and output directory in `build_training_batches.py` to match your local data structure.

### Load Pre-trained Weights

```python
import torch
import torch.nn as nn
from torchvision import models

# Initialize ResNet-50 architecture
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  # 6 ESKAPE classes

# Load trained weights (trained with DataParallel)
state_dict = torch.load('ESKAPe_Resnet.pth', map_location='cpu')
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Ready for inference
# Class order: 0=Eco, 1=Sau, 2=Kpn, 3=Aba, 4=Pae, 5=Efm
```

### Training

```
python ESKAPe_Resnet.py
```

> **Note:** Training requires your own dataset. Modify the data paths and class labels in `ESKAPe_Resnet.py` to match your local setup.

## Performance Summary

| Metric | Value |
|:-------|:------|
| Species-level classification accuracy | >92% |
| ESKAPe abundance quantification (mock mixtures) | >82% |
| Dominant pathogen identification (clinical samples) | >78% |
| Imaging-to-identification time | <10 min |
| Median time to diagnosis (with brief culture) | 5–6 h |

## Contact

For questions regarding the code or model, please open an issue on GitHub or contact the corresponding author.
