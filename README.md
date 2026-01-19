# AutoLabel Pro: Human-in-the-Loop Annotation with Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

**AutoLabel Pro** is an intelligent annotation tool designed to accelerate object detection labeling. It integrates **YOLO-World** for zero-shot prompting and an **Active Learning** loop that allows the model to learn from user corrections in real-time.

<p align="center">
  <img src="assets/demo.gif" width="800" alt="AutoLabel Pro Demo">
</p>

## 🌟 Key Features

- **Zero-Shot Annotation**: Powered by YOLO-World, label unseen objects using natural language prompts (e.g., "safety helmet", "smoke").
- **Recursive Training (Active Learning)**: The model trains in the background as you label. The more you label, the smarter it gets.
- **Model Agnostic**: Supports Ultralytics YOLO (v8/v11/World) and classic Torchvision models (Faster-RCNN, RetinaNet, SSD).
- **Format Support**: Seamless import/export for Pascal VOC (XML) and COCO (JSON).

## 🛠️ Installation

### Prerequisites
- Windows / Linux / macOS
- Python 3.8+
- CUDA (Optional, for GPU acceleration)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AutoLabel-Pro.git
   cd AutoLabel-Pro
