# ResNet TensorRT Benchmark

A learning-focused project for production inference optimization with TensorRT and quantization.

## 🎯 Learning Goals

By completing this project, you will understand:
- ONNX model export from PyTorch
- TensorRT engine building and optimization
- Quantization: FP32 → FP16 → INT8
- INT8 calibration process
- Latency vs throughput trade-offs

## ⚠️ Prerequisites

**TensorRT SDK Required** - Install before starting:
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- Requires NVIDIA GPU with Compute Capability 6.0+

## 📚 Learning Roadmap

### Phase 1: Theory (Notes)

| # | Note | Topic | Status |
|---|------|-------|--------|
| 0 | [00_tensorrt_overview.md](notes/00_tensorrt_overview.md) | TensorRT architecture and workflow | ⬜ |
| 1 | [01_quantization.md](notes/01_quantization.md) | FP32 vs FP16 vs INT8 trade-offs | ⬜ |
| 2 | [02_calibration.md](notes/02_calibration.md) | INT8 calibration process | ⬜ |
| 3 | [03_optimization_layers.md](notes/03_optimization_layers.md) | Layer fusion and optimizations | ⬜ |

### Phase 2: Model Export

| # | File | Concept | Status |
|---|------|---------|--------|
| 1 | [export_onnx.py](models/export_onnx.py) | PyTorch → ONNX conversion | ⬜ |

### Phase 3: TensorRT Implementation

| # | File | Concept | Prereq Notes | Status |
|---|------|---------|--------------|--------|
| 1 | [build_engine.cpp](src/build_engine.cpp) | TensorRT engine builder | 0 | ⬜ |
| 2 | [infer_fp32.cpp](src/infer_fp32.cpp) | FP32 inference | 0 | ⬜ |
| 3 | [infer_fp16.cpp](src/infer_fp16.cpp) | FP16 inference | 1 | ⬜ |
| 4 | [infer_int8.cpp](src/infer_int8.cpp) | INT8 with calibration | 1, 2 | ⬜ |

### Phase 4: Analysis

| # | File | Purpose | Status |
|---|------|---------|--------|
| 1 | [compare_precision.py](benchmarks/compare_precision.py) | Accuracy vs latency analysis | ⬜ |

## 📁 Directory Structure

```
resnet-tensorrt-bench/
├── README.md
├── Makefile
├── models/
│   ├── export_onnx.py           # PyTorch → ONNX export
│   └── resnet50.onnx            # Exported model (generated)
├── src/
│   ├── build_engine.cpp         # TensorRT engine builder
│   ├── infer_fp32.cpp           # FP32 inference
│   ├── infer_fp16.cpp           # FP16 inference
│   └── infer_int8.cpp           # INT8 with calibration
├── calibration/
│   ├── calibration_dataset/     # Sample images for INT8
│   └── int8_calibrator.cpp      # INT8 calibration class
├── notes/
│   ├── 00_tensorrt_overview.md
│   ├── 01_quantization.md
│   ├── 02_calibration.md
│   └── 03_optimization_layers.md
└── benchmarks/
    └── compare_precision.py
```

## 🔧 Build & Run (After TensorRT Installed)

```bash
# Export model
python models/export_onnx.py

# Build TensorRT examples
make all

# Run benchmarks
./bin/infer_fp32           # FP32 baseline
./bin/infer_fp16           # FP16 (~2x faster)
./bin/infer_int8           # INT8 (~4x faster, slight accuracy loss)
```

## 📊 Expected Results

| Precision | Latency | Throughput | Accuracy |
|-----------|---------|------------|----------|
| PyTorch FP32 | ~10ms | ~100 img/s | 100% |
| TensorRT FP32 | ~5ms | ~200 img/s | 100% |
| TensorRT FP16 | ~2.5ms | ~400 img/s | ~99.9% |
| TensorRT INT8 | ~1.5ms | ~670 img/s | ~99% |

## 🔗 Relevance

**Micron Interview**: Covers quantization (LoRA, INT8) from job requirements. Understanding precision trade-offs is essential for memory-efficient AI.

## 📖 References

- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [NVIDIA TensorRT Samples](https://github.com/NVIDIA/TensorRT)
- [Quantization Aware Training (QAT)](https://pytorch.org/docs/stable/quantization.html)
