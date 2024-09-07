

# 附录：

for QDQ documents and how tensorrt process QDQ nodes, pls ref our developer guide: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks
 And TensorRT provide a tool to do PTQ and QAT in pytorch: https://github.com/NVIDIA/TensorRT/blob/release/8.5/tools/pytorch-quantization/examples/torchvision/classification_flow.py
 Besides, our team develop a sample to guide how to got best perf on Yolov7: https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat
 And the QDQ best placement guide is here: https://github.com/NVIDIA-AI-IOT/yolo_deepstream/blob/main/yolov7_qat/doc/Guidance_of_QAT_performance_optimization.md