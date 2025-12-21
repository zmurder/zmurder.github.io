# 简介

经常使用很多的Nvidia的相关指令，这里做一个简单的汇总，包括trtexec、nsight_systems、DLA等相关的指令。方便后面查询使用

# trtexec常用指令

## trtexec构建engine

* 构建一个fp32精度的engine

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=test.onnx --saveEngine=test_fp32.engine --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_fp32_layer.json --exportProfile=test_fp32_profile.json --exportTimes=test_fp32_time.json  >test_fp32.log 2>&1
  
  ```

* 构建一个fp16精度的engine

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=test.onnx --saveEngine=test_fp16.engine --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_fp16_layer.json --exportProfile=test_fp16_profile.json --exportTimes=test_fp16_time.json  >test_fp16.log 2>&1
  
  ```

* 构建一个best精度的engine

  ```bash
  /usr/src/tensorrt/bin/trtexec --onnx=test.onnx --saveEngine=test_int8.engine --int8 --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_int8_layer.json --exportProfile=test_int8_profile.json --exportTimes=test_int8_time.json  >test_int8.log 2>&1
  
  ```

* 构建一个插入QDQ节点量化后的engine

  ```bash
  /usr/src/tensorrt/bin/trtexec  --onnx=test_ptq.onnx --saveEngine=test_ptq.engine  --int8 --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_ptq_layer.json --exportProfile=test_ptq_profile.json --exportTimes=test_ptq_time.json >test_ptq.log 2>&1
  
  ```

* 构建一个动态batch的engine，int8精度的engine

  ```bash
  trtexec --onnx=resnet34_batch.onnx  --verbose --dumpLayerInfo --dumpProfile  --minShapes=input:10x3x224x224 --optShapes=input:10x3x224x224 --maxShapes=input:10x3x224x224 --profilingVerbosity=detailed --int8 --saveEngine=resnet34_batch.onnxINT8.engine >resnet34_batch.onnxINT8.log 2>&1
  
  ```



```bash
#DLA 
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s_6x1024x768-lss_256x256-bs4.onnx --saveEngine=yolov8s_6x1024x768-lss_256x256-bs4_int8_dla.engine --int8 --useDLACore=0 --allowGPUFallback --verbose --dumpLayerInfo --dumpProfile --useSpinWait --separateProfileRun  >yolov8s_6x1024x768-lss_256x256-bs4_int8_dla.engine.log 2>&1


#qat
/usr/src/tensorrt/bin/trtexec  --onnx=test_ptq.onnx --saveEngine=test_ptq.engine  --int8 --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_ptq_layer.json --exportProfile=test_ptq_profile.json --exportTimes=test_ptq_time.json >test_ptq.log 2>&1

#fp16
/usr/src/tensorrt/bin/trtexec --onnx=test.onnx --saveEngine=test_fp16.engine --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_fp16_layer.json --exportProfile=test_fp16_profile.json --exportTimes=test_fp16_time.json  >test_fp16.log 2>&1

#fp32 
/usr/src/tensorrt/bin/trtexec --onnx=test.onnx --saveEngine=test_fp32.engine --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_fp32_layer.json --exportProfile=test_fp32_profile.json --exportTimes=test_fp32_time.json  >test_fp32.log 2>&1

#best 
/usr/src/tensorrt/bin/trtexec --onnx=test.onnx --saveEngine=test_int8.engine --int8 --fp16 --verbose --dumpLayerInfo --dumpProfile --profilingVerbosity=detailed  --exportLayerInfo=test_int8_layer.json --exportProfile=test_int8_profile.json --exportTimes=test_int8_time.json  >test_int8.log 2>&1
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

## 批量转换脚本

一个脚本实现文件夹内的所有onnx转换为对应的enigne

```bash
#!/bin/bash

# 帮助信息函数
show_help() {
    local exit_code=$1
    echo "Usage: $0 <mode> [onnx_dir]  eg:buildSplitOrin.sh deploy /mapdata/zyd/model/0820orin/10"
    echo "Available modes:"
    echo "  deploy      - 部署模式（启用特殊IO格式）"
    echo "  best        - 最佳模式（启用 fp16 + int8）"
    echo "  fp16        - FP16模式（仅启用 fp16）"
    echo "  fp32        - FP16模式（仅启用 fp32）"
    echo "  qdq         - QDQ模型（仅启用 fp16）"
    echo "  onnx_dir    - 包含 .onnx 文件的目录"
    echo "  -h, --help  - 显示此帮助信息"
    exit "$exit_code"
}

# 参数检查
if [ $# -eq 0 ]; then
    show_help 1
fi
# 参数解析
MODE="$1"

# 帮助参数处理
if [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    show_help 0
fi
# 模式合法性检查
valid_modes=("deploy" "best" "fp16" "qdq" "fp32")
valid=false
for mode in "${valid_modes[@]}"; do
    if [ "$mode" = "$MODE" ]; then
        valid=true
        break
    fi
done
if [ "$valid" = false ]; then
    echo "错误：无效模式 '$MODE'"
    show_help 1
fi

#目录参数
if [ $# -ge 2 ]; then
    ONNX_DIR="$2"
else
    echo "错误：未指定 onnx_dir"
    exit 1
fi
# 检查 ONNX_DIR 是否存在
if [ ! -d "$ONNX_DIR" ]; then
    echo "错误：ONNX 目录不存在: $ONNX_DIR"
    exit 1
fi
echo "ONNX_DIR: $ONNX_DIR"

# 动态后缀
suffix=""
if [ "$MODE" = "best" ]; then
    suffix="_best"
elif [ "$MODE" = "fp16" ]; then
    suffix="_fp16"
elif [ "$MODE" = "fp32" ]; then
    suffix="_fp32"
elif [ "$MODE" = "qdq" ]; then
    suffix="_qdq"
elif [ "$1" = "deploy" ]; then
    suffix="_deploy"
fi
# 指定包含 .onnx 文件的文件夹
# ONNX_DIR="/mapdata/zyd/model/0820orin/4"

# 遍历文件夹中的所有 .onnx 文件
for onnx_file in "$ONNX_DIR"/*.onnx; do
    # 获取不带扩展名的文件名
    base_name=$(basename "$onnx_file" .onnx)

    # 生成对应的 engine 名称、日志名称等
    engine_file="${ONNX_DIR}/${base_name}${suffix}.engine"
    layer_info_file="${ONNX_DIR}/${base_name}${suffix}_layer.json"
    profile_info_file="${ONNX_DIR}/${base_name}${suffix}_profile.json"
    time_info_file="${ONNX_DIR}/${base_name}${suffix}_time.json"
    log_file="${ONNX_DIR}/${base_name}${suffix}.log"

    echo "Processing $onnx_file, generated $engine_file, $layer_info_file, $time_info_file, and $log_file"

    # 如果是部署模式，启用特殊处理逻辑
    if [ "$MODE" = "deploy" ]; then
        flag=false
        # 判断文件名是否包含 'seg_head' 或 'com_deploy'
        if [[ "$base_name" == *"seg_head"* ]]; then
            echo "文件名包含 'seg_head'"
            # 执行分支1的命令
            # inputIOFormats="fp16:chw,fp16:chw" #seg_trans0,seg_trans1
            inputIOFormats="fp16:chw" #0820 seg_fuse_conv
            outputIOFormats="int32:chw" #seg_output
            flag=true
        elif [[ "$base_name" == *"com_deploy"* ]]; then
            echo "文件名包含 'com_deploy'"
            # 执行分支2的命令
            # inputIOFormats="fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw" # image_tensors,concat_ext_0,concat_ext_1,ext_input1,ext_input2
            inputIOFormats="fp16:chw,fp32:chw32,fp32:chw32,fp16:chw,fp16:chw" # image_tensors,concat_ext_0,concat_ext_1,ext_input1,ext_input2 fused dual gridSample plugin更新
            # outputIOFormats="fp16:chw,fp16:chw,fp16:chw" #vt_output,seg_trans0,seg_trans1
            # outputIOFormats="fp16:chw32,fp16:chw,fp16:chw" #vt_output,seg_trans0,seg_trans1  fused dual gridSample plugin更新
            outputIOFormats="fp16:chw,fp16:chw" #vt_output,0820 seg_fuse_conv
            flag=true
        elif [[ "$base_name" == *"concat_ext"* ]]; then
            echo "文件名包含 'concat_ext'"
            # 执行分支2的命令
            inputIOFormats="fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw" # cam2egos,post_rots,post_trans,bda_rot,intrins,distortions
            outputIOFormats="fp32:chw,fp32:chw" #concat_ext_0,concat_ext_1
            flag=true
        elif [[ "$base_name" == *"extupdate_deploy"* ]]; then
            echo "文件名包含 'extupdate_deploy'"
            # 执行分支2的命令
            inputIOFormats="fp16:chw,fp16:chw" # aug_matrix,sensor2ego
            outputIOFormats="fp16:chw,fp16:chw" #ext_input1,ext_input2
            flag=true
        elif [[ "$base_name" == *"sub2_s_deploy"* ]]; then
            echo "文件名包含 'sub2_s_deploy'"
            # 执行分支2的命令
            inputIOFormats="fp16:chw" # vt_output
            outputIOFormats="int32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw" 
            # occ_output det3d_head0_reg det3d_head0_height det3d_head0_dim det3d_head0_rot det3d_head0_heatmap
            flag=true
        fi
        if [ "$flag" = false ]; then
            echo "文件名 '$base_name' 不匹配任何模式，跳过处理"
            continue
        fi
    fi
    
    # precision_args=""
    # input_ioformat_args=""
    # output_ioformat_args=""
    # if [ "$MODE" = "best" ] || [ "$MODE" = "qdq" ]; then
    #     precision_args=" --fp16 --int8"
    # elif [ "$MODE" = "fp16" ]; then
    #     precision_args=" --fp16"
    # elif [ "$MODE" = "deploy" ]; then
    #     precision_args=" --fp16 --int8"
    #     input_ioformat_args="--inputIOFormats=\"$inputIOFormats\""
    #     output_ioformat_args="--outputIOFormats=\"$outputIOFormats\""
    # fi
        
    # 执行 trtexec 命令
    cmd=(
        /usr/src/tensorrt/bin/trtexec  \
            --onnx="$onnx_file" \
            --saveEngine="$engine_file" \
            --verbose \
            --maxAuxStreams=0 \
            --dumpLayerInfo \
            --dumpProfile \
            --separateProfileRun --tacticSources=-CUBLAS,-CUBLAS_LT,-CUDNN \
            --plugins=/mapdata/zyd/model/plugin/libgrid_sample_v2.1.so --plugins=/mapdata/zyd/model/plugin/libbevpool_v1.1.so --plugins=/mapdata/zyd/model/plugin/libfusedgridsampling.so --plugins=/mapdata/zyd/model/plugin/libfuse_dual_grid_sample_v2.1.so\
            --profilingVerbosity=detailed \
            --exportLayerInfo="$layer_info_file" \
            --exportProfile="$profile_info_file" \
            --exportTimes="$time_info_file" \
    )
    # 有条件地添加参数
    if [ "$MODE" = "best" ]; then
        echo "MODE='$MODE': 启用 --fp16 --int8"
        cmd+=( --fp16 )
        cmd+=( --int8 )
    elif [ "$MODE" = "qdq" ] || [ "$MODE" = "deploy" ]; then
        if [[ "$base_name" == *"sub2_s_deploy"* ]]; then
            echo "MODE='$MODE' with base_name='$base_name': 启用 --fp16 only (skip --int8)"
            cmd+=( --fp16 )
        elif [[ "$base_name" == *"_extupdate_deploy"* ]]; then
            echo "MODE='$MODE' with base_name='$base_name': 启用 --fp16"
            cmd+=( --fp16 )
        else
            echo "MODE='$MODE': 启用 --fp16 --int8"
            cmd+=( --fp16 )
            cmd+=( --int8 )
        fi
    elif [ "$MODE" = "fp16" ]; then
        echo "MODE='$MODE': 启用 --fp16"
        cmd+=( --fp16 )
    fi
    if [ "$MODE" = "deploy" ]; then
        echo "MODE='$MODE': 启用 --inputIOFormats 和 --outputIOFormats"
        cmd+=( --inputIOFormats="$inputIOFormats" )
        cmd+=( --outputIOFormats="$outputIOFormats" )
        # > "$log_file" 2>&1 
    fi
    "${cmd[@]}" > "$log_file" 2>&1       
    echo "Processed $onnx_file, generated $engine_file, $layer_info_file, $time_info_file, and $log_file"
done
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

## 划分子图

```bash
#划分两部分
#!/bin/bash

# 设置路径变量
MODEL_DIR=/home/zyd/code/onnx/test
MODEL_NAME=test.onnx
OUTPUT_DIR=$MODEL_DIR

# 拼接完整模型路径
MODEL_PATH=$MODEL_DIR/$MODEL_NAME


# #划分为两个onnx
# # 子图1输出文件
SUB1_OUTPUT=$OUTPUT_DIR/test_2d.onnx

# # 子图2输出文件
SUB2_OUTPUT=$OUTPUT_DIR/test_s_sub2.onnx

# 提取第一个子图
polygraphy surgeon extract $MODEL_PATH \
    --inputs image_tensors:[4,3,512,768]:float32 image_tensors:[20,280,280,2]:float32 depth_points:[20,280,280,2]:float32\
    --outputs 2621:float32 \
    -o $SUB1_OUTPUT
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