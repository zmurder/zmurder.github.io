# 1 简介

在之前的文章中`7-TensorRT中的INT8`介绍了TensorRT的量化理论基础，这里就根据理论实现相关的代码

# 2 PTQ

## 2.1 trtexec

 int8量化 使用`trtexec` 参数`--int8`来生成对应的`--int8`的`engine`，但是精度损失会比较大。也可使用int8和fp16混合精度同时使用`--fp16 --int8`

```bash
trtexec --onnx=XX.onnx --saveEngine=model.plan --int8
trtexec --onnx=XX.onnx --saveEngine=model.plan --int8 --fp16
```

上面的方式没有使用量化文件进行校正，因此精度损失非常多。可以使用`trtexec`添加`--calib=`参数指定`calibration file`文件进行校准。如下面的指令

**这个是我目前采用的方法**

```bash
trtexec --onnx=test.onnx --verbose --dumpLayerInfo --dumpProfile  --int8 --calib=calibratorfile_test.txt --saveEngine=test.onnx.INT8.trtmodel --exportProfile=build.log
```

上面指定的`--calib=calibratorfile_test.txt`是需要我们自己生成的，使用我们自己的数据集，在build engine阶段，tensorrt会根据数据分布和我们指定的校准器来生成的。如在`7-TensorRT中的INT8`中介绍的熵校准`Entropy calibration`使用的校准器就是`IInt8EntropyCalibrator2`。

将ONNX转换为INT8的TensorRT引擎，需要:

1. 准备一个校准集，用于在转换过程中寻找使得转换后的激活值分布与原来的FP32类型的激活值分布差异最小的阈值;
2. 并写一个校准器类，该类需继承trt.IInt8EntropyCalibrator2父类，并重写get_batch_size,   get_batch, read_calibration_cache, write_calibration_cache这几个方法。可直接使用`myCalibrator.py`，需传入图片文件夹地址

下面是具体的实现代码

### **2.1.1 python版本代码生成calibratorfile和int8engine**

参考：https://github.com/aiLiwensong/Pytorch2TensorRT

官方代码在`/samples/python/int8_caffe_mnist`

下面回答来自GPT

在 TensorRT 中，`trt.Builder` 是用于构建 TensorRT 引擎的主要类。当你使用 `trt.Builder` 来构建一个 TensorRT 引擎时，通过调用 `builder.build_engine` 函数来生成引擎。在执行这个函数时，如果你使用了 int8 量化，TensorRT 将会在量化过程中使用校准数据，并在量化过程中生成校准数据文件。

具体而言，当你调用 `builder.build_engine` 函数时，TensorRT 会根据设置执行 int8 量化过程，包括使用指定的校准方法和数据集来生成校准数据。在量化过程中，TensorRT 将会不断地调用 `IInt8Calibrator::getBatch` 函数来获取数据样本，并根据这些样本来执行量化。在执行量化过程时，TensorRT 会在合适的时机将校准数据写入到校准数据文件中。

`main.py`

```python
# main.py

import argparse
from trt_convertor import ONNX2TRT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch2TensorRT")
    # parser.add_argument("--dynamic", action='store_true', help='batch_size')  # not ok yet.
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    parser.add_argument("--channel", type=int, default=3, help='input channel')
    parser.add_argument("--height", type=int, default=384, help='input height')
    parser.add_argument("--width", type=int, default=640, help='input width')
    parser.add_argument("--cache_file", type=str, default='', help='cache_file')
    parser.add_argument("--mode", type=str, default='int8', help='fp32, fp16 or int8')
    parser.add_argument("--onnx_file_path", type=str, default='model.onnx', help='onnx_file_path')
    parser.add_argument("--engine_file_path", type=str, default='model.engine', help='engine_file_path')
    parser.add_argument("--imgs_dir", type=str, default='path_to_images_dir',
                        help='calibrator images dir')
    args = parser.parse_args()
    print(args)
    if args.mode.lower() == 'int8':
        # Note that: if use int8 mode, you should prepare a calibrate dataset and create a Calibrator class.
        # In Calibrator class, you should override 'get_batch_size, get_batch',
        # 'read_calibration_cache', 'write_calibration_cache'.
        # You can reference implementation of MyEntropyCalibrator.
        from myCalibrator import MyEntropyCalibrator
        calib = MyEntropyCalibrator(tensor_shape=(args.batch_size, args.channel, args.height, args.width),
                                    imgs_dir=args.imgs_dir)
    else:
        calib = None

    ONNX2TRT(args, calib=calib)
```

对应的`myCalibrator.py`

```python
#myCalibrator.py
# -*- coding: utf-8 -*-
"""
Created on : 20200608
@author: LWS

Create custom calibrator, use to calibrate int8 TensorRT model.

Need to override some methods of trt.IInt8EntropyCalibrator2, such as get_batch_size, get_batch,
read_calibration_cache, write_calibration_cache.

"""
import glob

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

import os
import numpy as np


class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, tensor_shape, imgs_dir,
                 mean=(0., 0., 0.), std=(255., 255., 255.)):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = 'CALIBRATOR.cache'
        self.mean = mean
        self.std = std

        self.batch_size, self.Channel, self.Height, self.Width = tensor_shape

        self.imgs = glob.glob(os.path.join(imgs_dir, "*"))
        np.random.shuffle(self.imgs)

        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.data_size = trt.volume([self.batch_size, self.Channel, self.Height, self.Width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size: \
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = cv2.imread(f)
                img = self.transform(img, (self.Width, self.Height))
                assert (img.nbytes == self.data_size / self.batch_size), 'not valid img!' + f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("\rbatch:[{}/{}]".format(self.batch_idx, self.max_batch_idx), end='')
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])

    def transform(self, img, img_size):
        w, h = img_size
        oh, ow = img.shape[0:2]
        s = min(w / ow, h / oh)
        nw, nh = int(round(ow * s)), int(round(oh * s))
        t, b, l, r = (h - nh) // 2, (h - nh + 1) // 2, (w - nw) // 2, (w - nw + 1) // 2
        if nw != ow or nh != oh:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
        img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=114)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - np.float32(self.mean)) * (1 / np.float32(self.std))
        img = img.transpose(2, 0, 1).copy()
        return img

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.Height * self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))

            return [int(self.device_input)]
        except Exception as e:
            print("get batch error: {}".format(e))
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


if __name__ == '__main__':
    calib = MyEntropyCalibrator(1, 3, 384, 640)
    batch = calib.get_batch(None)
    print(batch)
```

`trt_convertor.py`如下

```python
# trt_convertor.py
# -*- coding: utf-8 -*-
import json
import tensorrt as trt


def ONNX2TRT(args, calib=None):
    """ convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    """

    assert args.mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # TRT>=7.0中onnx解析器的network，需要指定EXPLICIT_BATCH
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:

        builder.max_batch_size = args.batch_size

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30

        # TODO: not ok yet.
        if args.dynamic:
            profile = builder.create_optimization_profile()
            profile.set_shape("images",
                              (1, args.channel, args.height, args.width),
                              (2, args.channel, args.height, args.width),
                              (4, args.channel, args.height, args.width)
                              )
            config.add_optimization_profile(profile)

        # builder.max_workspace_size = 1 << 30
        if args.mode.lower() == 'int8':
            assert (builder.platform_has_fast_int8 == True), "not support int8"
            assert (calib is not None), "need calib!"
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
        elif args.mode.lower() == 'fp16':
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            config.set_flag(trt.BuilderFlag.FP16)

        print('Loading ONNX file from path {}...'.format(args.onnx_file_path))
        with open(args.onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for e in range(parser.num_errors):
                    print(parser.get_error(e))
                raise TypeError("Parser parse failed.")

        print('Parsing ONNX file complete!')

        print('Building an engine from file {}; this may take a while...'.format(args.onnx_file_path))
        engine = builder.build_engine(network, config)
        if engine is not None:
            print("Create engine success! ")
        else:
            print("ERROR: Create engine failed! ")
            return

        # 保存计划文件
        print('Saving TRT engine file to path {}...'.format(args.engine_file_path))
        with open(args.engine_file_path, "wb") as f:
            # # Metadata
            # meta = json.dumps(self.metadata)
            # t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
            # t.write(meta.encode())
            f.write(engine.serialize())
        print('Engine file has already saved to {}!'.format(args.engine_file_path))

        return engine


def loadEngine2TensorRT(filepath):
    """
    通过加载计划文件，构建TensorRT运行引擎
    """
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    # 反序列化引擎
    with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine
```

### **2.1.2 c++代码版本生成calibratorfile和int8engine**

参考代码

*  **[Miacis](https://github.com/SakodaShintaro/Miacis)**  
* **[tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)**
* **[tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)**   
* https://www.cnblogs.com/TaipKang/p/15542329.html 这篇博客中也有全套的代码供参考。



下面代码来自    **[tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)**   

强烈推荐这个 **[tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)**   里面有很多教程。

```C
#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "trt_calibrator.hpp"
#include <string>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

namespace model{

Model::Model(string onnx_path, logger::Level level, Params params) {
    m_onnxPath      = onnx_path;
    m_workspaceSize = WORKSPACESIZE;
    m_logger        = make_shared<logger::Logger>(level);
    m_timer         = make_shared<timer::Timer>();
    m_params        = new Params(params);
    m_enginePath    = changePath(onnx_path, "../engine", ".engine", getPrec(params.prec));
}

void Model::load_image(string image_path) {
    if (!fileExists(image_path)){
        LOGE("%s not found", image_path.c_str());
    } else {
        m_imagePath = image_path;
        LOG("*********************INFERENCE INFORMATION***********************");
        LOG("\tModel:      %s", getFileName(m_onnxPath).c_str());
        LOG("\tImage:      %s", getFileName(m_imagePath).c_str());
        LOG("\tPrecision:  %s", getPrec(m_params->prec).c_str());
    }
}

void Model::init_model() {
    /* 一个model的engine, context这些一旦创建好了，当多次调用这个模型的时候就没必要每次都初始化了*/
    if (m_context == nullptr){
        if (!fileExists(m_enginePath)){
            LOG("%s not found. Building trt engine...", m_enginePath.c_str());
            build_engine();
        } else {
            LOG("%s has been generated! loading trt engine...", m_enginePath.c_str());
            load_engine();
        }
    }else{
        m_timer->init();
        reset_task();
    }
}

bool Model::build_engine() {
    // 我们也希望在build一个engine的时候就把一系列初始化全部做完，其中包括
    //  1. build一个engine
    //  2. 创建一个context
    //  3. 创建推理所用的stream
    //  4. 创建推理所需要的device空间
    // 这样，我们就可以在build结束以后，就可以直接推理了。这样的写法会比较干净
    auto builder       = shared_ptr<IBuilder>(createInferBuilder(*m_logger), destroy_trt_ptr<IBuilder>);
    auto network       = shared_ptr<INetworkDefinition>(builder->createNetworkV2(1), destroy_trt_ptr<INetworkDefinition>);
    auto config        = shared_ptr<IBuilderConfig>(builder->createBuilderConfig(), destroy_trt_ptr<IBuilderConfig>);
    auto parser        = shared_ptr<IParser>(createParser(*network, *m_logger), destroy_trt_ptr<IParser>);

    config->setMaxWorkspaceSize(m_workspaceSize);
    config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED); //这里也可以设置为kDETAIL;

    if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){
        return false;
    }

    if (builder->platformHasFastFp16() && m_params->prec == model::FP16) {
        config->setFlag(BuilderFlag::kFP16);
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    } else if (builder->platformHasFastInt8() && m_params->prec == model::INT8) {
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    }

    shared_ptr<Int8EntropyCalibrator> calibrator(new Int8EntropyCalibrator(
        64, 
        "calibration/calibration_list_coco.txt", 
        "calibration/calibration_table.txt",
        3 * 224 * 224, 224, 224));
    config->setInt8Calibrator(calibrator.get());

    auto engine        = shared_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config), destroy_trt_ptr<ICudaEngine>);
    auto plan          = builder->buildSerializedNetwork(*network, *config);
    auto runtime       = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);

    // 保存序列化后的engine
    save_plan(*plan);

    // 根据runtime初始化engine, context, 以及memory
    setup(plan->data(), plan->size());

    // 把优化前和优化后的各个层的信息打印出来
    LOGV("Before TensorRT optimization");
    print_network(*network, false);
    LOGV("After TensorRT optimization");
    print_network(*network, true);

    return true;
}

bool Model::load_engine() {
    // 同样的，我们也希望在load一个engine的时候就把一系列初始化全部做完，其中包括
    //  1. deserialize一个engine
    //  2. 创建一个context
    //  3. 创建推理所用的stream
    //  4. 创建推理所需要的device空间
    // 这样，我们就可以在load结束以后，就可以直接推理了。这样的写法会比较干净
    
    if (!fileExists(m_enginePath)) {
        LOGE("engine does not exits! Program terminated");
        return false;
    }

    vector<unsigned char> modelData;
    modelData     = loadFile(m_enginePath);
    
    // 根据runtime初始化engine, context, 以及memory
    setup(modelData.data(), modelData.size());

    return true;
}

void Model::save_plan(IHostMemory& plan) {
    auto f = fopen(m_enginePath.c_str(), "wb");
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
}

/* 
    可以根据情况选择是否在CPU上跑pre/postprocess
    对于一些edge设备，为了最大化GPU利用效率，我们可以考虑让CPU做一些pre/postprocess，让其执行与GPU重叠
*/
void Model::inference() {
    if (m_params->dev == CPU) {
        preprocess_cpu();
    } else {
        preprocess_gpu();
    }

    enqueue_bindings();

    if (m_params->dev == CPU) {
        postprocess_cpu();
    } else {
        postprocess_gpu();
    }
}


bool Model::enqueue_bindings() {
    m_timer->start_gpu();
    if (!m_context->enqueueV2((void**)m_bindings, m_stream, nullptr)){
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    m_timer->stop_gpu("trt-inference(GPU)");
    return true;
}

void Model::print_network(INetworkDefinition &network, bool optimized) {

    int inputCount = network.getNbInputs();
    int outputCount = network.getNbOutputs();
    string layer_info;

    for (int i = 0; i < inputCount; i++) {
        auto input = network.getInput(i);
        LOGV("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }

    for (int i = 0; i < outputCount; i++) {
        auto output = network.getOutput(i);
        LOGV("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }
    
    int layerCount = optimized ? m_engine->getNbLayers() : network.getNbLayers();
    LOGV("network has %d layers", layerCount);

    if (!optimized) {
        for (int i = 0; i < layerCount; i++) {
            char layer_info[1000];
            auto layer   = network.getLayer(i);
            auto input   = layer->getInput(0);
            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOGV("layer_info: %-40s:%-25s->%-25s[%s]", 
                layer->getName(),
                printTensorShape(input).c_str(),
                printTensorShape(output).c_str(),
                getPrecision(layer->getPrecision()).c_str());
        }

    } else {
        auto inspector = shared_ptr<IEngineInspector>(m_engine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            LOGV("layer_info: %s", inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}

string Model::getPrec(model::precision prec) {
    switch(prec) {
        case model::precision::FP16:   return "fp16";
        case model::precision::INT8:   return "int8";
        default:                       return "fp32";
    }
}

} // namespace model
```

src/model/calibrator.hpp代码如下

```c++
#include "NvInfer.h"
#include "trt_calibrator.hpp"
#include "utils.hpp"
#include "trt_logger.hpp"
#include "trt_preprocess.hpp"

#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace nvinfer1;

namespace model{

/*
 * calibrator的构造函数
 * 我们在这里把calibration所需要的数据集准备好，需要保证数据集的数量可以被batchSize整除
 * 同时由于calibration是在device上进行的，所以需要分配空间
 */
Int8EntropyCalibrator::Int8EntropyCalibrator(
    const int&    batchSize,
    const string& calibrationDataPath,
    const string& calibrationTablePath,
    const int&    inputSize,
    const int&    inputH,
    const int&    inputW):

    m_batchSize(batchSize),
    m_inputH(inputH),
    m_inputW(inputW),
    m_inputSize(inputSize),
    m_inputCount(batchSize * inputSize),
    m_calibrationTablePath(calibrationTablePath)
{
    m_imageList = loadDataList(calibrationDataPath);
    m_imageList.resize(static_cast<int>(m_imageList.size() / m_batchSize) * m_batchSize);
    std::random_shuffle(m_imageList.begin(), m_imageList.end(), 
                        [](int i){ return rand() % i; });
    CUDA_CHECK(cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float)));
}

/*
 * 获取做calibration的时候的一个batch的图片，之后上传到device上
 * 需要注意的是，这里面的一个batch中的每一个图片，都需要做与真正推理是一样的前处理
 * 这里面我们选择在GPU上进行前处理，所以处理万
 */
bool Int8EntropyCalibrator::getBatch(
    void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (m_imageIndex + m_batchSize >= m_imageList.size() + 1)
        return false;

    LOG("%3d/%3d (%3dx%3d): %s", 
        m_imageIndex + 1, m_imageList.size(), m_inputH, m_inputW, m_imageList.at(m_imageIndex).c_str());
    
    /*
     * 对一个batch里的所有图像进行预处理
     * 这里可有以及个扩展的点
     *  1. 可以把这个部分做成函数，以函数指针的方式传给calibrator。因为不同的task会有不同的预处理
     *  2. 可以实现一个bacthed preprocess
     * 这里留给当作今后的TODO
     */
    cv::Mat input_image;
    for (int i = 0; i < m_batchSize; i ++){
        input_image = cv::imread(m_imageList.at(m_imageIndex++));
        preprocess::preprocess_resize_gpu(
            input_image, 
            m_deviceInput + i * m_inputSize,
            m_inputH, m_inputW, 
            preprocess::tactics::GPU_BILINEAR_CENTER);
    }

    bindings[0] = m_deviceInput;

    return true;
}
    
/* 
 * 读取calibration table的信息来创建INT8的推理引擎, 
 * 将calibration table的信息存储到calibration cache，这样可以防止每次创建int推理引擎的时候都需要跑一次calibration
 * 如果没有calibration table的话就会直接跳过这一步，之后调用writeCalibrationCache来创建calibration table
 */
const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    void* output;
    m_calibrationCache.clear();

    ifstream input(m_calibrationTablePath, ios::binary);
    input >> noskipws;
    if (m_readCache && input.good())
        copy(istream_iterator<char>(input), istream_iterator<char>(), back_inserter(m_calibrationCache));

    length = m_calibrationCache.size();
    if (length){
        LOG("Using cached calibration table to build INT8 trt engine...");
        output = &m_calibrationCache[0];
    }else{
        LOG("Creating new calibration table to build INT8 trt engine...");
        output = nullptr;
    }
    return output;
}

/* 
 * 将calibration cache的信息写入到calibration table中
*/
void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    ofstream output(m_calibrationTablePath, ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}

} // namespace model
```

## 2.2 **python onnx转trt**

- **操作流程**：按照常规方案导出onnx，onnx序列化为tensorrt engine之前打开int8量化模式并采用校正数据集进行校正；
- **优点**：
  - 导出onnx之前的所有操作都为常规操作；
  -  相比在pytorch中进行PTQ int8量化，所需显存小；
- **缺点**：
  - 量化过程为黑盒子，无法看到中间过程；
  -  校正过程需在实际运行的tensorrt版本中进行并保存tensorrt  engine；
  - 量化过程中发现，即使模型为动态输入，校正数据集使用时也必须与推理时的输入shape[N, C, H,  W]完全一致，否则，效果非常非常差，动态模型慎用。
- 操作示例参看[onnx2trt_ptq.py](https://link.zhihu.com/?target=https%3A//github.com/Susan19900316/yolov5_tensorrt_int8/blob/master/onnx2trt_ptq.py)

下面的代码其实就是上面的2.1.1

```python
import tensorrt as trt
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2


def get_crop_bbox(img, crop_size):
    """Randomly get a crop bounding box."""
    margin_h = max(img.shape[0] - crop_size[0], 0)
    margin_w = max(img.shape[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    return crop_x1, crop_y1, crop_x2, crop_y2

def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    return img

class yolov5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, imgpath, batch_size, channel, inputsize=[384, 1280]):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = 'yolov5.cache'
        self.batch_size = batch_size
        self.Channel = channel
        self.height = inputsize[0]
        self.width = inputsize[1]
        self.imgs = [os.path.join(imgpath, file) for file in os.listdir(imgpath) if file.endswith('jpg')]
        np.random.shuffle(self.imgs)
        self.imgs = self.imgs[:2000]
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.calibration_data = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)
        # self.data_size = trt.volume([self.batch_size, self.Channel, self.height, self.width]) * trt.float32.itemsize
        self.data_size = self.calibration_data.nbytes
        self.device_input = cuda.mem_alloc(self.data_size)
        # self.device_input = cuda.mem_alloc(self.calibration_data.nbytes)

    def free(self):
        self.device_input.free()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.height * self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs)
            return [self.device_input]
        except:
            print('wrong')
            return None
    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size: \
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.height, self.width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = cv2.imread(f)  # BGR
                crop_size = [self.height, self.width]
                crop_bbox = get_crop_bbox(img, crop_size)
                # crop the image
                img = crop(img, crop_bbox)
                img = img.transpose((2, 0, 1))[::-1, :, :]  # BHWC to BCHW ,BGR to RGB
                img = np.ascontiguousarray(img)
                img = img.astype(np.float32) / 255.
                assert (img.nbytes == self.data_size / self.batch_size), 'not valid img!' + f
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            # os.fsync(f)


def get_engine(onnx_file_path, engine_file_path, cali_img, mode='FP32', workspace_size=4096):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    def build_engine():
        assert mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            explicit_batch_flag
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser:
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            config.max_workspace_size = workspace_size * (1024 * 1024)  # workspace_sizeMiB
            # 构建精度
            if mode.lower() == 'fp16':
                config.flags |= 1 << int(trt.BuilderFlag.FP16)

            if mode.lower() == 'int8':
                print('trt.DataType.INT8')
                config.flags |= 1 << int(trt.BuilderFlag.INT8)
                config.flags |= 1 << int(trt.BuilderFlag.FP16)
                calibrator = yolov5EntropyCalibrator(cali_img, 26, 3, [384, 1280])
                # config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
                config.int8_calibrator = calibrator
            # if True:
            #     config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, min=(1, 3, 384, 1280), opt=(12, 3, 384, 1280), max=(26, 3, 384, 1280))
            config.add_optimization_profile(profile)
            # config.set_calibration_profile(profile)
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            # plan = builder.build_serialized_network(network, config)
            # engine = runtime.deserialize_cuda_engine(plan)
            engine = builder.build_engine(network,config)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                # f.write(plan)
                f.write(engine.serialize())
            return engine
        
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main(onnx_file_path, engine_file_path, cali_img_path, mode='FP32'):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    get_engine(onnx_file_path, engine_file_path, cali_img_path, mode)


if __name__ == "__main__":
    onnx_file_path = '/home/models/boatdetect_yolov5/last_nms_dynamic.onnx'
    engine_file_path = "/home/models/boatdetect_yolov5/last_nms_dynamic_onnx2trtptq.plan"
    cali_img_path = '/home/data/frontview/test'
    main(onnx_file_path, engine_file_path, cali_img_path, mode='int8')
```



## 2.3 **polygraphy工具**

- **操作流程：**按照常规方案导出onnx，onnx序列化为tensorrt engine之前打开int8量化模式并采用校正数据集进行校正；
- **优点：**1. 相较于1.1，代码量更少，只需完成校正数据的处理代码；
- **缺点**：1
  - 同上所有; 
  -  动态尺寸时，校正数据需与–trt-opt-shapes相同
  - 内部默认最多校正20个epoch；
- 安装polygraphy

```text
pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

- 量化

```text
polygraphy convert XX.onnx --int8 --data-loader-script loader_data.py --calibration-cache XX.cache -o XX.plan --trt-min-shapes images:[1,3,384,1280] --trt-opt-shapes images:[26,3,384,1280] --trt-max-shapes images:[26,3,384,1280] #量化
```

- [loader_data.py](https://link.zhihu.com/?target=https%3A//github.com/Susan19900316/yolov5_tensorrt_int8/blob/master/loader_data.py)为较正数据集加载过程，自动调用脚本中的load_data()函数：

## 2.4 pytorch中执行(推荐)

实际上是使用`pytorch-quantization` PyTorch 中进行量化（Quantization）的库，支持PTQ和QAT的量化。这里给出就是PTQ的例子

> 注：在pytorch中执行导出的onnx将产生一个明确量化的模型，属于显示量化

- **操作流程：**安装pytorch_quantization库->加载校正数据->加载模型（在加载模型之前，启用quant_modules.initialize() 以保证原始模型层替换为量化层）->校正->导出onnx;

- **优点：**

  - 通过导出的onnx能够看到每层量化的过程；
  - onnx导出为tensort  engine时可以采用trtexec(注：命令行需加–int8，需要fp16和int8混合精度时，再添加–fp16，这里有些疑问，GPT说导出 ONNX 模型时进行了量化，那么在使用 `trtexec` 转换为 TensorRT Engine 时，你不需要添加任何特别的参数。因为 ONNX 模型中已经包含了量化后的信息，TensorRT 在转换过程中会自动识别并保留这些信息。因此不知道是不是需要`--int8`，我感觉不需要了。)，比较简单；
  -  pytorch校正过程可在任意设备中进行；
  - 相较上述方法，校正数据集使用shape无需与推理shape一致，也能获得较好的结果，动态输入时，推荐采用此种方式。

- **缺点：**导出onnx时，显存占用非常大；

- 操作示例参看：pytorch模型进行量化导出[yolov5_pytorch_ptq.py](https://link.zhihu.com/?target=https%3A//github.com/Susan19900316/yolov5_tensorrt_int8/blob/master/pytorch_yolov5_ptq.py)

  ```python
  import torch
  import torch.nn as nn
  import torch.optim as optim
  import torch.utils.data as data
  import torchvision.transforms as transforms
  from torchvision import models, datasets
  
  import pytorch_quantization
  from pytorch_quantization import nn as quant_nn
  from pytorch_quantization import quant_modules
  from pytorch_quantization import calib
  from tqdm import tqdm
  
  print(pytorch_quantization.__version__)
  
  import os
  import tensorrt as trt
  import numpy as np
  import time
  import wget
  import tarfile
  import shutil
  import cv2
  import random
  
  from models.yolo import Model
  from models.experimental import End2End
  
  def compute_amax(model, **kwargs):
      # Load calib result
      for name, module in model.named_modules():
          if isinstance(module, quant_nn.TensorQuantizer):
              if module._calibrator is not None:
                  if isinstance(module._calibrator, calib.MaxCalibrator):
                      module.load_calib_amax()
                  else:
                      module.load_calib_amax(**kwargs)
      model.cuda()
  
  def collect_stats(model, data_loader):
      """Feed data to the network and collect statistics"""
      # Enable calibrators
      for name, module in model.named_modules():
          if isinstance(module, quant_nn.TensorQuantizer):
              if module._calibrator is not None:
                  module.disable_quant()
                  module.enable_calib()
              else:
                  module.disable()
  
      # Feed data to the network for collecting stats
      for i, image in tqdm(enumerate(data_loader)):
          model(image.cuda())
  
      # Disable calibrators
      for name, module in model.named_modules():
          if isinstance(module, quant_nn.TensorQuantizer):
              if module._calibrator is not None:
                  module.enable_quant()
                  module.disable_calib()
              else:
                  module.enable()
  
  def get_crop_bbox(img, crop_size):
      """Randomly get a crop bounding box."""
      margin_h = max(img.shape[0] - crop_size[0], 0)
      margin_w = max(img.shape[1] - crop_size[1], 0)
      offset_h = np.random.randint(0, margin_h + 1)
      offset_w = np.random.randint(0, margin_w + 1)
      crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
      crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
      return crop_x1, crop_y1, crop_x2, crop_y2
  
  def crop(img, crop_bbox):
      """Crop from ``img``"""
      crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
      img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
      return img
  
  class CaliData(data.Dataset):
      def __init__(self, path, num, inputsize=[384, 1280]):
          self.img_files = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('jpg')]
          random.shuffle(self.img_files)
          self.img_files = self.img_files[:num]
          self.height = inputsize[0]
          self.width = inputsize[1]
  
      def __getitem__(self, index):
          f = self.img_files[index]
          img = cv2.imread(f)  # BGR
          crop_size = [self.height, self.width]
          crop_bbox = get_crop_bbox(img, crop_size)
          # crop the image
          img = crop(img, crop_bbox)
          img = img.transpose((2, 0, 1))[::-1, :, :]  # BHWC to BCHW ,BGR to RGB
          img = np.ascontiguousarray(img)
          img = img.astype(np.float32) / 255.
          return img
  
      def __len__(self):
          return len(self.img_files)
  
  
  if __name__ == '__main__':
      pt_file = 'runs/train/exp/weights/best.pt'
      calib_path = 'XX/train'
      num = 2000 # 用来校正的数目
      batchsize = 4
      # 准备数据
      dataset = CaliData(calib_path, num)
      dataloader = data.DataLoader(dataset, batch_size=batchsize)
  
      # 模型加载
      quant_modules.initialize() #保证原始模型层替换为量化层
      device = torch.device('cuda:0')
      ckpt = torch.load(pt_file, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
      # QAT
      q_model = ckpt['model']
      yaml = ckpt['model'].yaml
      q_model = Model(yaml, ch=yaml['ch'], nc=yaml['nc']).to(device)  # creat
      q_model.eval()
      q_model = End2End(q_model).cuda()
      ckpt = ckpt['model']
      modified_state_dict = {}
      for key, val in ckpt.state_dict().items():
          # Remove 'module.' from the key names
          if key.startswith('module'):
              modified_state_dict[key[7:]] = val
          else:
              modified_state_dict[key] = val
      q_model.model.load_state_dict(modified_state_dict)
  
  
      # Calibrate the model using calibration technique.
      with torch.no_grad():
          collect_stats(q_model, dataloader)
          compute_amax(q_model, method="entropy")
  
      # Set static member of TensorQuantizer to use Pytorch’s own fake quantization functions
      quant_nn.TensorQuantizer.use_fb_fake_quant = True
  
      # Exporting to ONNX
      dummy_input = torch.randn(26, 3, 384, 1280, device='cuda')
      input_names = ["images"]
      output_names = ["num_dets", 'det_boxes']
      # output_names = ['outputs']
      save_path = '/'.join(pt_file.split('/')[:-1])
      onnx_file = os.path.join(save_path, 'best_ptq.onnx')
      dynamic = dict()
      dynamic['images'] = {0: 'batch'}
      dynamic['num_dets'] = {0: 'batch'}
      dynamic['det_boxes'] = {0: 'batch'}
      torch.onnx.export(
          q_model,
          dummy_input,
          onnx_file,
          verbose=False,
          opset_version=13,
          do_constant_folding=False,
          input_names=input_names,
          output_names=output_names,
          dynamic_axes=dynamic)
  ```

  上面的代码，生成的 ONNX 模型是已经量化过的。以下是代码中的量化过程：

  1. **导入 PyTorch Quantization 库**：
     - 通过 `import pytorch_quantization` 以及其他相关模块的导入，使用了 PyTorch Quantization 库中的功能。
  2. **量化模型**：
     - 在加载模型后，执行了量化操作。在 `__main__` 中，通过 `collect_stats` 和 `compute_amax` 函数执行了量化统计和计算最大值的操作。这是典型的 QAT（Quantization Aware Training）过程，其中使用校准数据集来估计量化参数。
     - 在执行 `compute_amax` 函数时，传递了 `method="entropy"` 参数，这表明使用的是熵方法来计算量化参数。
     - 最后，通过 `torch.onnx.export` 函数将量化后的模型导出为 ONNX 格式。

# 3 QAT

实际上是使用`pytorch-quantization` PyTorch 中进行量化（Quantization）的库，支持PTQ和QAT的量化。这里给出就是QAT的例子

> 注：在pytorch中执行导出的onnx将产生一个明确量化的模型，属于显式量化

- **操作流程：**安装pytorch_quantization库->加载训练数据->加载模型（在加载模型之前，启用quant_modules.initialize() 以保证原始模型层替换为量化层）->训练->导出onnx;
- **优点：**1. 模型量化参数重新训练，训练较好时，精度下降较少； 2. 通过导出的onnx能够看到每层量化的过程；2. onnx导出为tensort  engine时可以采用trtexec(注：命令行需加–int8，需要fp16和int8混合精度时，再添加–fp16)，比较简单；3.训练过程可在任意设备中进行；
- **缺点：**1.导出onnx时，显存占用非常大；2.最终精度取决于训练好坏；3. QAT训练shape需与推理shape一致才能获得好的推理结果；4. 导出onnx时需采用真实的图片输入作为输入设置
- 操作示例参看[yolov5_pytorch_qat.py](https://link.zhihu.com/?target=https%3A//github.com/Susan19900316/yolov5_tensorrt_int8/blob/master/pytorch_yolov5_qat.py)感知训练，参看[export_onnx_qat.py](https://link.zhihu.com/?target=https%3A//github.com/Susan19900316/yolov5_tensorrt_int8/blob/master/onnx2trt_ptq.py)

```python
import torch
import pytorch_quantization
from pytorch_quantization import nn as quant_nn

print(pytorch_quantization.__version__)

import os
import numpy as np
from models.experimental import End2End



if __name__ == '__main__':
    pt_file = 'runs/train/exp/weights/best.pt'

    # 模型加载
    device = torch.device('cuda:0')
    ckpt = torch.load(pt_file, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    q_model = ckpt['model']
    q_model.eval()
    q_model = End2End(q_model).cuda().float()


    # Set static member of TensorQuantizer to use Pytorch’s own fake quantization functions
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Exporting to ONNX
    # dummy_input = torch.randn(26, 3, 384, 1280, device='cuda')
    im = np.load('im.npy') # 重要：真实图像
    dummy_input = torch.from_numpy(im).cuda()
    dummy_input = dummy_input.float()
    dummy_input = dummy_input / 255
    input_names = ["images"]
    output_names = ['num_dets', 'det_boxes']
    save_path = '/'.join(pt_file.split('/')[:-1])
    onnx_file = os.path.join(save_path, 'best_nms_dynamic_qat.onnx')
    dynamic = {'images': {0: 'batch'}}
    dynamic['num_dets'] = {0: 'batch'}
    dynamic['det_boxes'] = {0: 'batch'}
    torch.onnx.export(
        q_model,
        dummy_input,
        onnx_file,
        verbose=False,
        opset_version=13,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic)
```



# 附录：

* [tensorrt官方int8量化方法汇总](https://zhuanlan.zhihu.com/p/648877516) 
* 参考代码：python版本 onnx转int8 trtengine https://github.com/aiLiwensong/Pytorch2TensorRT/tree/master
* 参考代码 **[Miacis](https://github.com/SakodaShintaro/Miacis)**  
* 参考代码v**[tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)**
* **强烈推荐参考代码** **[tensorrt_starter](https://github.com/kalfazed/tensorrt_starter)**   
*  强烈推荐知乎的一个文章，包含PTQ和QAT的流程和python代码实现**[yolov5_tensorrt_int8](https://github.com/Susan19900316/yolov5_tensorrt_int8)**     
* 参考代码 https://www.cnblogs.com/TaipKang/p/15542329.html
* [pytorch-quantization’s documentation](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#pytorch-quantization-s-documentation)