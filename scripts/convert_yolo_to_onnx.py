#!/usr/bin/env python3
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path
import onnxruntime as ort

def convert_yolo_to_onnx(pytorch_model_path: str, output_path: str = None, imgsz: int = 640):
    # Load the PyTorch model
    print(f"Loading PyTorch YOLO model from: {pytorch_model_path}")
    model = YOLO(pytorch_model_path)
    
    # Set output path if not provided
    if output_path is None:
        pytorch_path = Path(pytorch_model_path)
        output_path = pytorch_path.parent / f"{pytorch_path.stem}.onnx"
    
    print(f"Converting to ONNX format...")
    print(f"Output path: {output_path}")
    print(f"Input image size: {imgsz}x{imgsz}")
    
    # Export to ONNX
    success = model.export(
        format='onnx',
        imgsz=imgsz,
        optimize=False,
        half=False,
        int8=False,
        dynamic=False,
        simplify=True,
        opset=11,
    )
    
    if success:
        print(f"✅ Successfully converted to ONNX: {output_path}")
        return str(output_path)
    else:
        print("❌ Failed to convert to ONNX")
        return None

def test_onnx_model(onnx_path: str, test_image_size: tuple = (640, 640)):
    try:
        
        print(f"Testing ONNX model: {onnx_path}")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Get input details
        input_info = session.get_inputs()[0]
        print(f"Input name: {input_info.name}")
        print(f"Input shape: {input_info.shape}")
        print(f"Input type: {input_info.type}")
        
        # Get output details
        output_info = session.get_outputs()[0]
        print(f"Output name: {output_info.name}")
        print(f"Output shape: {output_info.shape}")
        print(f"Output type: {output_info.type}")
        
        # Create dummy input
        dummy_input = np.random.rand(1, 3, *test_image_size).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_info.name], {input_info.name: dummy_input})
        
        print(f"ONNX model test successful!")
        print(f"Output shape: {outputs[0].shape}")
        
        return True
    except Exception as e:
        print(f"ONNX model test failed: {e}")
        return False

if __name__ == "__main__":
    pytorch_model = "notebooks/yolo/best_so_far.pt"
    
    print("=== YOLO PyTorch to ONNX Conversion ===")
    
    # Convert to ONNX
    onnx_path = convert_yolo_to_onnx(pytorch_model)
    
    if onnx_path:
        # Test the ONNX model
        print("\n=== Testing ONNX Model ===")
        test_onnx_model(onnx_path)
    
    print("\n=== Conversion Complete ===")