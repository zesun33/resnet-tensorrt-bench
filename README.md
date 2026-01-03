# ResNet50 Inference "Speed Run" (TensorRT)

## Goal
Optimize a standard ResNet50 model for production inference, benchmarking the speedup gained from TensorRT compilation and reduced precision (FP16/INT8).

## Roadmap
- [ ] **01_export_onnx.py**: Export a pre-trained ResNet50 from PyTorch to ONNX.
- [ ] **02_build_engine.py**: Use `tensorrt` Python API to build an optimized engine (testing FP16 and INT8 flags).
- [ ] **03_inference_bench.py**: Measure throughput (img/sec) and latency (ms) for:
    - PyTorch Eager Mode
    - PyTorch `torch.compile`
    - TensorRT (FP32)
    - TensorRT (FP16/INT8)

## References
*   [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
*   [Deep Learning Deployment with TensorRT](https://developer.nvidia.com/tensorrt)
