# 🧠 Video Captioning Project
This repository contains a Video-to-Text generation system designed to synthesize descriptive natural language captions from raw video input. By leveraging a BLIP (Bootstrapping Language-Image Pre-training) architecture fine-tuned on the MSR-VTT dataset, the model bridges the gap between temporal visual features and semantic linguistic descriptions.

## 🏗️ Pipeline Architecture

```mermaid
graph LR
    %% Styling Classes
    classDef data fill:#ffe0b2,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef train fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef deploy fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000;

    %% Workflow Nodes
    subgraph Phase 1: Data
    A[📁 MSR-VTT 2k Subset]:::data --> B(🎞️ Extract 8 Uniform Frames):::data
    B --> C[(💾 Save as .npz Tensors)]:::data
    end

    subgraph Phase 2: BLIP Fine-Tuning
    C --> D{🎲 Dynamic: Sample 1 Frame}:::train
    D --> E[🧊 Freeze Vision Encoder]:::train
    E --> F[🔥 Train Text Decoder]:::train
    end

    subgraph Phase 3: Inference
    F --> G[📊 Evaluate Metrics]:::deploy
    G --> H((🌐 Gradio Web App)):::deploy
    end

    %% Flow adjustments for compactness
    style Phase 1 fill:none,stroke:#f57c00,stroke-dasharray: 5 5
    style Phase 2 fill:none,stroke:#388e3c,stroke-dasharray: 5 5
    style Phase 3 fill:none,stroke:#1976d2,stroke-dasharray: 5 5
```

## 🚀 Features
- **Video Captioning**: The project uses a pre-trained BLIP model to generate captions for videos.
- **Frame Sampling**: The project includes a frame sampling functionality to extract frames from videos at uniform intervals.
- **Model Training**: The project involves training a video captioning model using the MSR-VTT dataset.
- **Gradio Application**: The project sets up a Gradio application to provide a user interface for interacting with the video captioning model.

## 🛠️ Tech Stack
- **Frontend**: Gradio
- **Backend**: PyTorch
- **AI Tools**: Transformers, BLIP model

- **Dependencies**: 
  - `gradio` for creating the web application
  - `torch` for deep learning operations
  - `transformers` for the BLIP model and its processor
  - `opencv-python` for frame sampling
  - `datasets` for loading the MSR-VTT dataset
  - `pandas` for data manipulation and analysis
  - `numpy` for numerical operations


## 💻 Usage
To run the project, follow these steps:
1. Clone the repository: `git clone https://github.com/your-repo/video-captioning.git`
2. Navigate to the project directory: `cd video-captioning`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Gradio application: `app.py`
   
You can try the live interface here:
**[Launch Gradio App](https://2c2ed0636868a383ba.gradio.live/)** *(Note: Public Gradio links expire after 72 hours. If the link is down, follow the local setup instructions below.)*

---

##
📂 Project Structure
```text
video-captioning/
├── .gitignore
├── Readme.md
├── requirements.txt
├── app.py
├── baseline_evaluation_report.txt
├── basemodel_evaluation.ipynb
├── dataset.ipynb
├── final_evaluation_report.txt
├── finalmodel_evaluation.ipynb
├── frame_sampling.ipynb
├── msrvtt_2k_preprocessed.csv
├── msrvtt_train_2k_fullcaptions.csv
├── test.ipynb
└── training.ipynb
```
## 📂 Dataset & Preprocessing

### The MSR-VTT Dataset
This project utilizes the **MSR-VTT** (Microsoft Research Video to Text) dataset, a large-scale benchmark for video understanding. While the original dataset contains 10,000 video clips, this project uses a curated subset of **2,000 videos** to optimize training time while maintaining a diverse vocabulary and visual range.

The initial dataset was structured in a CSV with the following core columns:
* `video_id`: A unique identifier for the clip.
* `captions`: The ground-truth textual descriptions.
* `video_path`: The local file path to the raw video file.

### 🎞️ Frame Sampling & Preprocessing
Raw video files (.mp4) are massive and computationally expensive to feed directly into a transformer model. To solve this, we implemented a robust **Frame Sampling** pipeline using OpenCV and NumPy:

1. **Uniform Temporal Sampling**: The script calculates the total frame count and extracts exactly **8 evenly spaced frames** across the video's duration to capture the complete action.
2. **Visual Standardization**: Each frame is converted to RGB and resized to **224x224 pixels**, matching the BLIP vision encoder's expected input.
3. **Tensor Formatting**: The 8 frames are stacked and transposed into the shape `(8, 3, 224, 224)` representing `(Frames, Channels, Height, Width)`.
4. **Efficient Storage (.npz)**: These processed arrays are saved as compressed NumPy files (`.npz`), drastically reducing I/O bottlenecks during training compared to raw videos.
5. **Automated Cleaning**: The script skips broken or overly short videos and outputs a clean `msrvtt_2k_preprocessed.csv` mapping captions directly to the new `.npz` files.

**💡 Note on Frame Usage:** While the preprocessing pipeline saves 8 frames per video, the PyTorch `Dataset` dynamically samples just **1 random frame** from this set during each training step. This strategy provides temporal variance across multiple epochs without the heavy memory overhead of processing full 3D video tensors.

## 🏋️ Training Overview
The model was fine-tuned on Apple Silicon (MPS) using a Vision-Frozen Strategy. By freezing the BLIP vision encoder and training only the text decoder, we achieved efficient, high-quality caption generation.

## 📉 Loss Convergence
The model showed steady optimization over 3 epochs, nearly cutting the initial loss by 35%.

### 📉 Training Loss

| Epoch | Average Loss |
| :---: | :---: |
| **1** | 3.0709 |
| **2** | 2.4607 |
| **3** | 2.0260 |

## ⚙️ Optimization Highlights
Frozen Vision Encoder: Reduced trainable parameters to focus on language generation.

Dynamic Frame Sampling: Picks 1 random frame per video each step to improve generalization.

Memory Efficient: Utilizes torch.mps.empty_cache() to maintain performance on Mac hardware.

## 📊 Evaluation Results

The model was evaluated against a baseline using standard language modeling metrics (BLEU, ROUGE-L, and CIDEr). The final model shows a substantial leap in performance across all categories.

| Metric | Baseline Score | Final Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Bleu_1** | 70.21 | **89.50** | +19.29 |
| **Bleu_2** | 50.34 | **81.08** | +30.74 |
| **Bleu_3** | 33.74 | **72.10** | +38.36 |
| **Bleu_4** | 22.36 | **62.99** | +40.63 |
| **ROUGE_L** | 47.86 | **73.08** | +25.22 |
| **CIDEr** | 29.92 | **87.39** | **+57.47** |


### 📈 Performance Analysis
* **Contextual Accuracy**: The **CIDEr** score saw the most significant jump (+57.47), indicating the final model is much better at capturing the specific consensus of the video content rather than just matching common words.
* **Structural Fluency**: The massive increase in **Bleu_4** (+40.63) shows that the model has progressed from generating simple fragments to high-quality, continuous 4-gram sequences.
* **Overall Recall**: A **ROUGE_L** score of 73.08 suggests the generated captions maintain high structural similarity to the ground truth references.



