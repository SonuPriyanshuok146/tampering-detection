# AI-Based Digital Forensic Image Tampering Detection

AI-based system that detects and localizes tampered regions in digital images by combining Convolutional Neural Networks (CNNs) with frequency-domain forensic techniques, and explains its decisions using Grad-CAM heatmaps.

## Overview

Digital images are widely used as evidence in journalism, legal proceedings, insurance claims, and social media verification. With accessible editing tools like Photoshop and GIMP, images can be tampered with — through splicing, copy-move forgery, or retouching — in ways invisible to the human eye.

This project detects such tampering by combining:
- **CNN-based classification** (transfer learning on EfficientNet/ResNet)
- **Frequency-domain analysis** (DCT-based Error Level Analysis) to catch compression inconsistencies
- **Grad-CAM explainability** so predictions come with a visual heatmap, not just a label

## Objectives

- Classify images as authentic or tampered (splicing / copy-move forgery)
- Detect compression inconsistencies using DCT-based Error Level Analysis
- Localize tampered regions via segmentation (stretch goal)
- Provide Grad-CAM heatmaps for interpretable predictions
- Evaluate on standard forensic benchmarks; test robustness to re-compression and resizing
- Serve predictions through a prototype web interface

## Methodology

```
Dataset (CASIA v2 / Columbia Splicing)
    → Preprocessing (ELA + DCT feature extraction)
    → CNN Feature Extraction (EfficientNet/ResNet backbone, transfer learning)
    → Classification Head (Authentic vs. Tampered)
    → Grad-CAM Explainability Layer
    → Tampering Report (prediction, confidence score, heatmap)
    → Web Interface
```

Training happens in two stages: a baseline classifier on ELA-preprocessed images first, then DCT frequency-domain statistics are fused in as an auxiliary input to improve robustness against more sophisticated forgeries. Data augmentation (JPEG re-compression, resizing, noise) simulates real-world image-sharing conditions throughout.

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, Pillow |
| Explainability | grad-cam (Grad-CAM) |
| Web Interface | Flask / FastAPI |
| Dev Tools | VS Code, Jupyter Notebook, Git |
| Datasets | CASIA v2, Columbia Image Splicing Dataset, COVERAGE |


## Setup

```bash
# 1. Create and activate environment
conda create -p venv python==3.10 -y
conda activate venv/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the project package (editable mode)
pip install -e .
```

## Datasets

Not included in this repository. Download separately and place under a local `data/` folder (ignored by Git):
- [CASIA v2](https://github.com/namtpham/casia2groundtruth)
- [Columbia Image Splicing Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm)
- COVERAGE

## Expected Outcomes

- Trained CNN classifier targeting >90% accuracy on benchmark test sets
- Working prototype highlighting suspected tampered regions
- Grad-CAM explainability layer for trust and interpretability
- Comparative evaluation report across forgery types and datasets

## Applications

- Digital forensics and law enforcement investigation support
- Journalism and media fact-checking
- Insurance claim image verification
- Social media content moderation
- Legal and court evidence authentication

## References

- Zhou, P., et al. "Learning Rich Features for Image Manipulation Detection." CVPR, 2018.
- Wu, Y., et al. "ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries." CVPR, 2019.
- Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV, 2017.
- CASIA Image Tampering Detection Dataset — National Laboratory of Pattern Recognition, China.
- [OpenCV Documentation](https://docs.opencv.org)
- [PyTorch Documentation](https://pytorch.org/docs)