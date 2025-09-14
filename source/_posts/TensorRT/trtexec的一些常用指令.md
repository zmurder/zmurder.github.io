# 简介

经常使用很多的Nvidia的相关指令，这里做一个简单的汇总，包括trtexec、nsight_systems、DLA等相关的指令。方便后面查询使用

# trtexec常用指令

## trtexec构建engine

* 构建一个fp32精度的engine

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=deformable_detr_refine.onnx --saveEngine=deformable_detr_refine_fp32.engine --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=deformable_detr_refine_fp32_layer.json --exportProfile=deformable_detr_refine_fp32_profile.json --exportTimes=deformable_detr_refine_fp32_time.json  >deformable_detr_refine_fp32.log 2>&1
  
  ```

* 构建一个fp16精度的engine

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=slot_20250311_inner_corner_sigma1.onnx --saveEngine=slot_20250311_inner_corner_sigma1_fp16.engine --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=slot_20250311_inner_corner_sigma1_fp16_layer.json --exportProfile=slot_20250311_inner_corner_sigma1_fp16_profile.json --exportTimes=slot_20250311_inner_corner_sigma1_fp16_time.json  >slot_20250311_inner_corner_sigma1_fp16.log 2>&1
  
  ```

* 构建一个best精度的engine

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=parking_3d_0310_orin_1_s.onnx --saveEngine=parking_3d_0310_orin_1_s_int8.engine --int8 --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=parking_3d_0310_orin_1_s_int8_layer.json --exportProfile=parking_3d_0310_orin_1_s_int8_profile.json --exportTimes=parking_3d_0310_orin_1_s_int8_time.json  >parking_3d_0310_orin_1_s_int8.log 2>&1
  
  ```

* 构建一个插入QDQ节点量化后的engine

  ```bash
  /usr/src/tensorrt/bin/trtexec  --onnx=test_ptq.onnx --saveEngine=test_ptq.engine  --int8 --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_ptq_layer.json --exportProfile=test_ptq_profile.json --exportTimes=test_ptq_time.json >test_ptq.log 2>&1
  
  ```

* 构建一个动态batch的engine，int8精度的engine

  ```bash
  trtexec --onnx=resnet34_batch.onnx  --verbose --dumpLayerInfo --dumpProfile  --minShapes=input:10x3x224x224 --optShapes=input:10x3x224x224 --maxShapes=input:10x3x224x224 --profilingVerbosity=detailed --int8 --saveEngine=resnet34_batch.onnxINT8.engine >resnet34_batch.onnxINT8.log 2>&1
  
  ```

## tetexec加载engine

```bash
trtexec --loadEngine=apa_lidarfreespace_u2nettp_v6_20240628_int8_0711.engine  --warmUp=5000  --iterations=2000 --verbose --dumpProfile  --dumpLayerInfo --profilingVerbosity=detailed
```



加载engine并执行输入的尺寸：

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=apa_LidarOD_centerpoint-pillar_v0_20241007_s_FP16.engine --shapes=voxels:20000x20x4,num_points:20000,coors:20000x4
 --warmUp=0 --duration=0 --iterations=200 
```



指定输入：

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=od4h_20250822.engine --loadInputs=image_tensors:img_cuda_ptr.bin,feat_points:feat_points.bin,depth_points:depth_points.bin --shapes=image_tensors:4x3x512x768
```



# DLA相关指令

## 转换一个onnx文件为DLA运行的engine

注意DLA只支持int8与fp16精度

```bash
trtexec --onnxtest.onnx --saveEnginetest.engine --int8 --useDLACore=0 --allowGPUFallback --verbose --dumpLayerInfo --dumpProfile --useSpinWait --separateProfileRun  >Lidar_u2nettp20240731.engine.log 2>&1

```

## 转dla控制层的精度，进而控制一层不在dla上运行



```bash
trtexec --onnx=test.onnx --saveEngine=test_layerPercision_dla.engine  --int8 --useDLACore=0 --allowGPUFallback --verbose --dumpLayerInfo --dumpProfile --useSpinWait --separateProfileRun --precisionConstraints=obey --layerPrecisions="/Concat":fp32  >test_layerPercision_dla.engine.log 2>&1

```

## DLA与QAT

参考：https://docs.nvidia.com/tao/tao-toolkit/text/qat_and_amp_for_training.html

如上面链接提到的

**DLA对量化参数的强依赖性**
 DLA硬件在设计上要求所有算子（包括卷积、激活函数等）必须显式指定量化参数（`scale`和`zero_point`），无法通过外部缓存文件或运行时动态计算补全缺失参数。这与GPU推理不同，后者允许通过TensorRT的量化缓存机制动态推导部分层的量化参数。

**Q/DQ节点覆盖不全的后果**
 若QAT模型存在未插入Q/DQ节点的层（例如某些激活函数或残差连接），DLA将无法推断这些层的`scale`值，导致推理失败或精度崩溃

因此我们如果是一个QDQ后的onnx，转换为DLA的engin需要下面的步骤：

### 从qdq的onnx生成onnx和cache文件

参考https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/blob/main/tools/qdq-translator/README.md

使用nvidia官方提供的代码，结合自己的QDQ的onnx文件生成一个类似PTQ的校准文件。将 QAT 图（量化感知训练的结果，包含量化和反量化 （Q/DQ） 节点）中的 ONNX 模型转换为没有 Q/DQ 节点的 ONNX 图，以及提取的张量尺度。

注意：这里的QDQ的onnx需要是每一层都插入QDQ节点的，否则会有很多的节点fallback到gpu上，影响效率

```bash
cuDLA-samples-main$ python export/qdq_translator/qdq_translator.py --input_onnx_models=./models/Lidar/v9_nearestqdq.onnx --output_dir=data/model/ > ./models/Lidar/v9_nearestqdq.onnx_qdq_translator.log 2>&1
```

### 构建一个经过qdq_translator后的noqdq onnx 和cache文件 生成DLA的engine

```bash
trtexec --onnx=v9_nearestqdq_noqdq.onnx --calib=v9_nearestqdq_precision_config_calib.cache --useDLACore=0 --int8 --fp16 --allowGPUFallback --saveEngine=v9_nearestqdq_noqdq.onnx_dlaFP16INT8.engine >v9_nearestqdq_noqdq.onnx_dlaFP16INT8.engine.log 2>&1

trtexec --onnx=v9_nearestqdq_noqdq.onnx --calib=v9_nearestqdq_precision_config_calib.cache --useDLACore=0 --int8  --allowGPUFallback --saveEngine=v9_nearestqdq_noqdq.onnx_dlaINT8.engine  >v9_nearestqdq_noqdq.onnx_dlaINT8.engine.log 2>&1
```

因为每一层都插入了QDQ节点，会带来精度误差，如果找到了敏感曾，希望控制某一层的精度为fp32（会Fallback到GPU上），使用下面的指令

```bash
trtexec --onnx=v9_nearestqdq_noqdq.onnx --calib=v9_nearestqdq_precision_config_calib.cache --useDLACore=0 --fp16 --allowGPUFallback --saveEngine=v9_nearestqdq_noqdq.onnx_dlaINT8_layerPercision.engine --precisionConstraints=obey --layerPrecisions="/Concat":fp32 >v9_nearestqdq_noqdq.onnx_dlaINT8_layerPercision.log 2>&1
```

## nsys 分析 dla

使用nsys可以分析一下在DLA上运行的情况，包括是不是有很多的fakkbcak到GPU

```bash
/algdata/zyd/nsight_systems/2022.3.2/target-linux-tegra-armv8/nsys  profile --accelerator-trace=nvmedia --trace=cuda,nvtx,cublas,cudla,cusparse,cudnn,nvmedia --force-overwrite true -o /algdata/zyd/DLATest/Lidar/apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit_layerPercision_dla.engine trtexec --loadEngine=/algdata/zyd/DLATest/Lidar/apa_lidarfreespace_u2nettp_6xdownsample_v9_20240810_scale_simlified_edit_layerPercision_dla.engine --iterations=10 --idleTime=50 --duration=0 --useSpinWait
```



# nsight_systems指令

## 使用nsys指令和trtexec结合分析engine推理

```bash
/algdata/zyd/nsight_systems/2022.3.2/target-linux-tegra-armv8/nsys  profile --stats=true --force-overwrite true -o /algdata/zyd/log/zydProfile4 --trace=cuda,cudnn,cublas,osrt,nvtx --gpuctxsw=true --gpu-metrics-device=0 trtexec --streams=1 --loadEngine=/algdata/zyd/profToolOrin/models/FSD/apa_freespace_fish_pid_v5_20240530_sim.onnxFP32.engine --warmUp=0 --duration=0 --iterations=100 --idleTime=100

```

## nsys  start stop的使用

```bash
/algdata/zyd/target-linux-tegra-armv8/nsys  launch --session-new=zydseesion --trace=cuda,osrt /algdata/apa/scripts/hpc/runHpcAllRestart.sh
/algdata/zyd/target-linux-tegra-armv8/nsys  start --session=zydseesion --stats=true --force-overwrite true -o /algdata/zyd/log/zydProfile_HPC17
/algdata/zyd/target-linux-tegra-armv8/nsys  stop --session=zydseesion
```

# polygraphy指令

## **转换模型并生成TensorRT引擎**

```bash
polygraphy convert test.onnx --int8 --trt-outputs mark all --calibration-cache test.onnx.cache -o test.onnx.polygraphy.int8.cache.engine
```



转换engine时指定某一层的精度：

```bash
/home/yezihua/anaconda3/envs/yezihua/bin/polygraphy run test.onnx  --trt --onnxrt --atol 0.5 --rtol 0.5 --fp16  --trt-outputs mark all --onnx-outputs mark all --precision-constraints prefer  --layer-precisions Mul_435:float32
```



## **检查生成的TensorRT引擎**

显示其所有层的信息(`--show layers`

```bash
polygraphy inspect model test.onnx.polygraphy.int8.cache.engine --model-type engine --show layers
```

## 计算onnx与engine 的误差

### 对比onnx与fp16精度的engine误差

对比每一层的误差：

```bash
/home/yezihua/anaconda3/envs/yezihua/bin/polygraphy run test.onnx  --trt --onnxrt --atol 0.5 --rtol 0.5 --fp16  --trt-outputs mark all --onnx-outputs mark all
```

### 对比onnx与现有的engine精度误差

前面的使用fp16精度的对比onnx误差（使用polygraphy转换出来的engine），一般较为常见的场景是， 直接转换成engine进行测试， 只有在精度异常时才会用到`polygrahy`工具。这时的engine不一定是fp32、fp16、int8精度的，也就是之前已经生成好的engine。如何对比现有的engine与onnx的误差呢？

**构建engine**

注意这里的 --trt-outputs mark all 表示保存各个层的输出， 可以通过指定层名称来保存对应的输出

```bash
polygraphy convert test.onnx --int8  --trt-outputs mark all --calibration-cache test.onnx.cache  -o test.onnx.polygraphy.int8.cache.engine
```

**使用ONNX Runtime运行模型并保存输入输出**

```bash
polygraphy run test.onnx --onnxrt --onnx-outputs mark all --save-inputs inputs.json --save-outputs run_0_outputs.json
```

**使用TensorRT引擎运行并验证结果**

使用的是之前生成的TensorRT引擎(`--trt`和`--model-type=engine`)，加载之前保存的输入(`--load-inputs inputs.json`)和预期输出(`--load-outputs run_0_outputs.json`)，并计算误差统计(`--check-error-stat median`)以验证结果准确性。

这里的test.onnx.polygraphy.int8.cache.engine是自己生成的一个engine

```bash
polygraphy run test.onnx.polygraphy.int8.cache.engine --trt --model-type=engine --check-error-stat median --trt-outputs mark all --load-inputs inputs.json --load-outputs run_0_outputs.json
```



trtexec --onnx=test.onnx --saveEngine=test_dla.engine --int8 --useDLACore=0 --allowGPUFallback --verbose --dumpLayerInfo --dumpProfile --useSpinWait --separateProfileRun >test_dla.engine.log 2>&1
 \#转dla控制层的精度，进而控制一层不在dla上运行
 trtexec --onnx=test.onnx --saveEngine=test_layerPercision_dla.engine --int8 --useDLACore=0 --allowGPUFallback --verbose --dumpLayerInfo --dumpProfile --useSpinWait --separateProfileRun --precisionConstraints=obey --layerPrecisions="/Concat":fp32 >test_layerPercision_dla.engine.log 2>&1
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
 /algdata/zyd/nsight_systems/2022.3.2/target-linux-tegra-armv8/nsys profile --accelerator-trace=nvmedia --trace=cuda,nvtx,cublas,cudla,cusparse,cudnn,nvmedia --force-overwrite true -o /algdata/zyd/DLATest/Lidar/test_layerPercision_dla.engine trtexec --loadEngine=/algdata/zyd/DLATest/Lidar/test_layerPercision_dla.engine --iterations=10 --idleTime=50 --duration=0 --useSpinWait

 https://github.com/NVIDIA-AI-IOT/cuDLA-samples/blob/main/export/README.md