# FloorPlan Object Detection (YOLOv8 Only)

A modular, production-ready object detection system for architectural floorplans, built with PyTorch and YOLOv8.

## Features
- YOLOv8-only, clean and focused codebase
- Modular structure for easy incremental development
- Robust config and logging
- CPU-friendly: designed for training/fine-tuning on low-resource machines (batch size 1–2, num_workers=0)
- Evaluation metrics: mAP, precision, recall, F1
- SOTA comparison and discussion (YOLOv8 vs. YOLOv5, Faster-RCNN, SSD)

## Getting Started
1. Clone the repo
2. Activate your conda env: `conda activate DL_CV`
3. Install requirements: `pip install -r requirements.txt`
4. Follow the step-by-step commits for incremental development

## Training on CPU
- Set `batch_size` to 1 or 2 in `config/config.json`
- Set `num_workers` to 0 in data loader
- Use pre-trained weights for fine-tuning (recommended)
- Expect slower training, but all features work on CPU

## Evaluation Metrics
- **mAP@0.5**: Main metric for object detection
- **Precision/Recall/F1**: Analyze trade-off between false positives/negatives
- **Speed (inference time)**: For real-time applications (reported, not optimized for CPU)
- **Memory usage**: Discussed for deployment

## SOTA Comparison (Discussion)
- **YOLOv8**: Best trade-off for speed and accuracy, especially for geometric objects in floorplans
- **YOLOv5**: Previous generation, still strong but less accurate than YOLOv8
- **Faster-RCNN**: High accuracy, slower inference
- **SSD**: Fast, but lower accuracy

*This repo is focused on YOLOv8 only for clarity, maintainability, and best results on architectural floorplans.* 