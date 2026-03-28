import torch
import cv2
import numpy as np
import os
import time
import gradio as gr
from transformers import BlipForConditionalGeneration, AutoProcessor
from onnxruntime import InferenceSession

# --- Configuration & Paths ---
BASE_PATH = "/Users/ayraj/Desktop/video_captioning"
MODEL_DIR = os.path.join(BASE_PATH, "blip_video_model_2") # The "Best Results" model
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
ONNX_PATH = os.path.join(BASE_PATH, "blip_vision_quantized.onnx")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Global State to prevent redundant loading ---
current_model = None
current_processor = None
current_blocks = None
onnx_session = None

class ONNXVisionWrapper:
    """Task 1: Hybrid Wrapper to trick PyTorch into using ONNX vision weights."""
    def __init__(self, session, config):
        self.session = session
        self.config = config

    def __call__(self, pixel_values):
        # Convert torch tensor to numpy for ONNX
        pixel_values_np = pixel_values.cpu().numpy().astype(np.float32)
        
        # Run ONNX inference
        outputs = self.session.run(None, {"pixel_values": pixel_values_np})
        last_hidden_state = torch.from_numpy(outputs[0]).to(DEVICE)
        
        # Return object compatible with HuggingFace Base model
        from collections import namedtuple
        Output = namedtuple("BaseModelOutput", ["last_hidden_state"])
        return Output(last_hidden_state=last_hidden_state)

def load_system(blocks, use_onnx):
    """Dynamic model loader handling Pruning (Task 3) and ONNX (Task 1)."""
    global current_model, current_processor, current_blocks, onnx_session
    
    # Skip loading if configuration hasn't changed
    if current_blocks == blocks and current_model is not None:
        return current_model, current_processor

    print(f"🔄 Loading System: {blocks} Blocks | ONNX: {use_onnx}")
    
    # Load Processor
    if current_processor is None:
        current_processor = AutoProcessor.from_pretrained(MODEL_DIR)
    
    # Load Model Base
    model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)
    
    # 1. Apply Architectural Pruning (Task 3)
    if blocks < 12:
        model.vision_model.encoder.layers = model.vision_model.encoder.layers[:blocks]
        ckpt = os.path.join(CHECKPOINT_DIR, f"tuned_{blocks}blocks.pt")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False)
            print(f"✅ Loaded pruned weights: {ckpt}")

    # 2. Apply ONNX Acceleration (Task 1)
    if use_onnx:
        if onnx_session is None and os.path.exists(ONNX_PATH):
            onnx_session = InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        
        if onnx_session:
            model.vision_model = ONNXVisionWrapper(onnx_session, model.vision_model.config)
            print("⚡ ONNX Acceleration Active")

    model.to(DEVICE)
    model.eval()
    
    current_model = model
    current_blocks = blocks
    return current_model, current_processor

def process_video(video_path, num_frames):
    """Task 2 & 4: Temporal processing of video stream."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select equidistant frames for temporal averaging
    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames = []
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    
    # Return averaged tensor (Motion Blur Trail)
    return np.mean(frames, axis=0).astype(np.uint8)

def inference(video, blocks, frames, use_onnx):
    """Final Generation Pipeline."""
    start_time = time.time()
    
    # 1. Load the specific model variation
    model, processor = load_system(int(blocks), use_onnx)
    load_time = time.time() - start_time
    
    # 2. Process Video
    process_start = time.time()
    avg_frame = process_video(video, int(frames))
    process_time = time.time() - process_start
    
    # 3. Generate Caption
    gen_start = time.time()
    inputs = processor(images=avg_frame, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_length=25, num_beams=3)
        caption = processor.decode(out[0], skip_special_tokens=True)
    
    gen_time = time.time() - gen_start
    total_time = time.time() - start_time
    
    # Latency Breakdown for Task 5 Monitoring
    metrics = {
        "Config Loading": f"{load_time:.3f}s",
        "Video Processing": f"{process_time:.3f}s",
        "AI Generation": f"{gen_time:.3f}s",
        "Total Latency": f"{total_time:.3f}s"
    }
    
    return caption, metrics

# --- Gradio UI Design ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎥 Production Video Captioning Dashboard")
    gr.Markdown("### Integrated Benchmarking: ONNX, Ablation, and Fine-Tuning Optimization")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Upload Source Video")
            
            with gr.Accordion("Architectural Variations (Task 3)", open=True):
                block_select = gr.Dropdown(
                    choices=["12", "10", "8", "6"], 
                    value="12", 
                    label="Vision Transformer Blocks (Pruning)"
                )
                frame_select = gr.Slider(
                    minimum=4, maximum=16, step=4, 
                    value=8, 
                    label="Temporal Sampling (Frame Count)"
                )
            
            with gr.Accordion("Backend Optimization (Task 1)", open=True):
                onnx_toggle = gr.Checkbox(label="Enable ONNX INT8 Acceleration", value=False)
            
            run_btn = gr.Button("🚀 Generate Caption", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Generated Caption", interactive=False)
            output_metrics = gr.JSON(label="Performance Metrics (Latency Profile)")
            
            gr.Markdown("""
            **Deployment Note:**
            - **12 Blocks:** Best semantic accuracy.
            - **6 Blocks:** Maximum speed for edge devices.
            - **ONNX Active:** Uses Task 1 optimized weights.
            """)

    run_btn.click(
        fn=inference, 
        inputs=[input_video, block_select, frame_select, onnx_toggle], 
        outputs=[output_text, output_metrics]
    )

if __name__ == "__main__":
    app.launch()