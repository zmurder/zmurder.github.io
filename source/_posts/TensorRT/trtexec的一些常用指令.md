trtexec --onnx=apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit.onnx --saveEngine=apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit_dla.engine --int8 --useDLACore=0 --allowGPUFallback --verbose --dumpLayerInfo --dumpProfile --useSpinWait --separateProfileRun >apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit_dla.engine.log 2>&1
 \#转dla控制层的精度，进而控制一层不在dla上运行
 trtexec --onnx=apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit.onnx --saveEngine=apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit_layerPercision_dla.engine --int8 --useDLACore=0 --allowGPUFallback --verbose --dumpLayerInfo --dumpProfile --useSpinWait --separateProfileRun --precisionConstraints=obey --layerPrecisions="/Concat":fp32 >apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit_layerPercision_dla.engine.log 2>&1
 trtexec --onnx=v9_nearest.onnx --saveEngine=v9_nearest_layerPercision_dla.engine --int8 --useDLACore=0 --allowGPUFallback --verbose --dumpLayerInfo --dumpProfile --useSpinWait --separateProfileRun --precisionConstraints=obey --layerPrecisions="/Concat":fp32 >v9_nearest_layerPercision_dla.engine.log 2>&1
 \#构建一个经过qdq_translator后的noqdq onnx 和cache文件 生成DLA的engine
 trtexec --onnx=v9_nearestqdq_noqdq.onnx --calib=v9_nearestqdq_precision_config_calib.cache --useDLACore=0 --int8 --fp16 --allowGPUFallback --saveEngine=v9_nearestqdq_noqdq.onnx_dlaFP16INT8.engine >v9_nearestqdq_noqdq.onnx_dlaFP16INT8.engine.log 2>&1
 trtexec --onnx=v9_nearestqdq_noqdq.onnx --calib=v9_nearestqdq_precision_config_calib.cache --useDLACore=0 --int8 --allowGPUFallback --saveEngine=v9_nearestqdq_noqdq.onnx_dlaINT8.engine >v9_nearestqdq_noqdq.onnx_dlaINT8.engine.log 2>&1
 trtexec --onnx=v9_nearestqdq_noqdq.onnx --calib=v9_nearestqdq_precision_config_calib.cache --useDLACore=0 --int8 --fp16 --allowGPUFallback --saveEngine=v9_nearestqdq_noqdq.onnx_dlaFP16INT8_layerPercision.engine --precisionConstraints=obey --layerPrecisions="/Concat":fp32 >v9_nearestqdq_noqdq.onnx_dlaFP16INT8_layerPercision.engine.log 2>&1
 trtexec --onnx=v9_nearestqdq_noqdq.onnx --calib=v9_nearestqdq_precision_config_calib.cache --useDLACore=0 --fp16 --allowGPUFallback --saveEngine=v9_nearestqdq_noqdq.onnx_dlaINT8_layerPercision.engine --precisionConstraints=obey --layerPrecisions="/Concat":fp32 >v9_nearestqdq_noqdq.onnx_dlaINT8_layerPercision.log 2>&1

 \#精度对比
 \#Lidar trtexec 给定输入 
 trtexec --loadEngine=v9_nearest_layerPercision_dla.engine --loadInputs='input':lidar_preBin.bin --exportOutput=Lidar_output_dla.json --dumpOutput
 CheckDLAOutputScripts$ python3 check_outputs_diff.py Lidar_output_FP32.json Lidar_output_dla.json 0
 CheckDLAOutputScripts$ python3 check_cosine_sim.py Lidar_output_FP32.json Lidar_output_dla.json 0

 \#nsys 分析 dla
 /algdata/zyd/nsight_systems/2022.3.2/target-linux-tegra-armv8/nsys profile --accelerator-trace=nvmedia --trace=cuda,nvtx,cublas,cudla,cusparse,cudnn,nvmedia --force-overwrite true -o /algdata/zyd/DLATest/Lidar/apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit_layerPercision_dla.engine trtexec --loadEngine=/algdata/zyd/DLATest/Lidar/apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit_layerPercision_dla.engine --iterations=10 --idleTime=50 --duration=0 --useSpinWait

 https://github.com/NVIDIA-AI-IOT/cuDLA-samples/blob/main/export/README.md