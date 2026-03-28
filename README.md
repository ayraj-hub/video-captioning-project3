# 🎥 Video Captioning: Production Edge Deployment & Interpretability (Project 3)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Optimized-ee4c2c.svg)
![ONNX](https://img.shields.io/badge/ONNX-INT8_Quantized-005ced.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00.svg)

Welcome to **Project 3** of the Video Captioning series. This repository transforms a heavy, research-grade Vision-Language Model (Salesforce BLIP) into a fast, interpretable, and robust system ready for edge deployment. 

Through systematic ablation, ONNX acceleration, domain adaptation, and Explainable AI (XAI) mapping, this project bridges the gap between theoretical deep learning and production software engineering.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Project Roadmap (The 5 Tasks)](#-project-roadmap)
3. [Interactive Dashboard (`app2.py`)](#-interactive-dashboard)
4. [Repository Structure](#-repository-structure)
5. [Key Achievements](#-key-achievements)
6. [Quick Start](#-quick-start)

---

## 🏗️ System Architecture

*The following diagram illustrates our hybrid PyTorch-ONNX inference pipeline. (Renders natively on GitHub!)*

```mermaid
graph TD
    A[Input Video .mp4] --> B[Frame Sampler]
    B -->|User selects 4-16 frames| C{Backend Optimization Engine}
    
    C -->|Task 1: ONNX INT8| D[Quantized C++ Vision Graph]
    C -->|Task 3: Pruned PyTorch| E[Ablated Vision Encoder <br> 6 / 8 / 10 / 12 Blocks]
    
    D --> F[Latent Visual Embeddings]
    E --> F
    
    F -->|Task 2: XAI Extraction| G[Cross-Attention Bridge]
    G -->|Extracts Heatmaps| H((Explainable AI Dashboard))
    
    G --> I[Autoregressive Text Decoder]
    I --> J[Generated Temporal Caption]
    
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#f96,stroke:#333,stroke-width:2px
