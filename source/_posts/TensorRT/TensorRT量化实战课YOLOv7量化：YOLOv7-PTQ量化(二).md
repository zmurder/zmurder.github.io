 

# 目录

*   *   [注意事项](#_2)
    *   [一、2023/11/19更新](#20231119_4)
    *   [二、2023/12/27更新](#20231227_8)
    *   [前言](#_11)
    *   [1\. YOLOv7-PTQ量化流程](#1_YOLOv7PTQ_24)
    *   [2\. 模型标定](#2__74)
    *   [3\. 敏感层分析](#3__505)
    *   [4\. PTQ量化](#4_PTQ_1078)
    *   [总结](#_1596)

# 注意事项

### 一、2023/11/19更新

**新增敏感层分析和 PTQ 量化代码工程化**

### 二、2023/12/27更新

**和 `贝蒂小熊` 看官交流的过程中发现模型标定小节中的一些描述存在问题，修改模型标定小节一些描述话语，重新梳理下 PTQ 量化和 QAT 量化的区别，具体可参考第 2 小节修改的内容**

# 前言

> 手写 AI 推出的全新 TensorRT 模型量化实战课程，[链接](https://www.bilibili.com/video/BV1NN411b7HZ/?spm_id_from=333.999.0.0)。记录下个人学习笔记，仅供自己参考。
> 
> 该实战课程主要基于手写 AI 的 Latte 老师所出的 [TensorRT下的模型量化](https://www.bilibili.com/video/BV18L41197Uz/)，在其课程的基础上，所整理出的一些实战应用。
> 
> 本次课程为 YOLOv7 量化实战第三课，主要介绍 YOLOv7-PTQ 量化
> 
> 课程大纲可看下面的思维导图

![b03a269e88c866011bdced6d9b002fee](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(二)/b03a269e88c866011bdced6d9b002fee.png)

# 1\. YOLOv7-PTQ量化流程

> 在上节课程中我们介绍了 YOLOv7-PTQ 量化中 QDQ 节点的插入，这节课我们将会完成 PTQ 模型的量化和导出。
> 
> 从上面的思维导图我们可以看到 YOLOv7-PTQ 量化的步骤，我们代码的讲解和编写都是按照这个流程来的。

在编写代码开始之前我们还是再来梳理下整个 YOLOv7-PTQ 量化的过程，如下：

## **1.** **准备工作**

首先是我们的准备工作，我们需要下载 YOLOv7 官方代码和预训练模型以及 COCO 数据集，并编写代码完成模型和数据的加载工作。

## **2.** **插入 QDQ 节点**

第二个就是我们需要对模型插入 QDQ 节点，它有以下两种方式：

*   **自动插入**
    *   使用 quant\_modules.initialize() 自动插入量化节点
*   **手动插入**
    *   使用 quant\_modules.initialize() 初始化量化操作或使用 QuantDescriptor() 自定义初始化量化操作
    *   编写代码为模型插入量化节点

## **3.** **标定**

第三部分就是我们的标定，其流程如下：

*   **1.** 通过将标定数据送到网络并收集网络每个层的输入输出信息
*   **2.** 根据统计出的信息，计算动态范围 range 和 scale，并保存在 QDQ 节点中

## **4.** **敏感层分析**

第四部分是敏感层分析，大致流程如下：

*   **1.** 进行单一逐层量化，只开启某一层的量化其他层都不开启
*   **2.** 在验证集上进行模型精度测试
*   **3.** 选出前 10 个对模型精度影响比较大的层，关闭这 10 个层的量化，在前向计算时使用 float16 而不去使用 int8

## **5.** **导出 PTQ 模型**

第五个就是我们在标定之后需要导出 PTQ 模型，导出流程如下：

*   **1.** 需要将我们上节课所说的 quant\_nn.TensorQuantizer.use\_fb\_fake\_quant 属性设置为 true
*   **2.** torch.onnx.export() 导出 ONNX 模型

## **6.** **性能对比**

第六个就是性能的对比，包括精度和速度的对比。

上节课我们完成了 YOLOv7-PTQ 量化流程中的准备工作和插入 QDQ 节点，这节我们继续按照流程走，先来实现模型的标定工作，让我们开始吧！！！🚀🚀🚀

# 2\. 模型标定

模型量化校准主要是由以下三个[函数](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782?utm_source=glcblog&spm=1001.2101.3001.7020)完成的：

## **1.** **calibrate\_model**

```python
def calibrate_model(model, dataloader, device):

    # 收集前向信息
    collect_stats(model, dataloader, device)

    # 获取动态范围，计算 amax 值，scale 值
    compute_amax(model, method = 'mse')
```

该函数主要是讲两个校准步骤组合起来，用于模型的整体校准，整体步骤如下：

*   使用 collect\_stats 函数收集前向传播的统计信息
*   调用 compute\_amax 函数计算量化的尺度因子 amax

## **2.** **collect\_stats**

```python
def collect_stats(model, data_loader, device, num_batch = 200):
    model.eval()

    # 开启校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # test
    with torch.no_grad():
        for i, datas in enumerate(data_loader):
            imgs = datas[0].to(device, non_blocking=True).float() / 255.0
            model(imgs)

            if i >= num_batch:
                break
    
    # 关闭校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
```

该函数的目的是收集模型在给定数据集上的激活统计信息，这通常是模型量化校准过程中的第一步，具体步骤如下：

*   设置模型为 eval 模型，确保不启用如 dropout 这样的训练特有的行为
*   遍历模型的所有模块，对于每一个 TensorQuantizer 实例
    *   如果有校准器存在，则禁用量化（不对输入进行量化）并启动校准模式（收集统计信息）
    *   如果没有校准器，则完全禁用该量化器（不执行任何操作）
*   使用 data\_loader 来提供数据，并通过模型执行前向传播
    *   讲数据转移到 device 上，并进行适当的归一化
    *   对每个批次数据，模型进行推理，但不进行梯度计算
    *   收集激活统计信息直到处理指定数量的批次
*   最后，遍历模型的所有模块，对于每一个 TensorQuantizer 实例
    *   如果有校准器存在，则启用量化并禁用校准模式
    *   如果没有校准器，则重新启用该量化器

## **3.** **compute\_amax**

```python
def compute_amax(model, **kwargs):
    
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                module._amax = module._amax.to(device)
```

一旦收集了激活的统计信息，该函数就会计算量化的尺度因子 amax（动态范围的最大值），这通常是模型量化校准过程中的第二步，步骤如下：

*   遍历模型的所有模块，对于每一个 TensorQuantizer 实例
    *   如果有校准器存在，则根据收集的统计信息计算 amax 值，这个值代表了激活的最大幅值，用于确定量化的尺度
    *   将 amax 值转移到 device 上，以便在后续中使用

下面我们简单总结下模型量化校准的流程：

*   **1.数据准备**: 准备用于标定的数据集，通常是[模型训练](https://ml-summit.org/cloud-member?uid=c1041&spm=1001.2101.3001.7020)或验证数据集的一个子集。
    
*   **2.收集统计信息**: 通过 collect\_stats 函数进行前向传播，以收集模型各层的激活分布统计信息。
    
*   **3.计算 amax**: 使用 compute\_amax 函数基于收集的统计信息计算量化参数（如最大激活值 amax）。
    

通过上述步骤，模型就可以得到合适的量化参数，从而在量化后保持性能并减小精度损失。

完整的示例代码如下：

```python
import os
import yaml
import test
import torch
import collections
from pathlib import Path
from models.yolo import Model
from pytorch_quantization import calib
from absl import logging as quant_logging
from utils.datasets import create_dataloader
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules import _utils as quant_nn_utils

def load_yolov7_model(weight, device='cpu'):
    ckpt  = torch.load(weight, map_location=device)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model

def prepare_val_dataset(cocodir, batch_size=32):
    dataloader = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader

def prepare_train_dataset(cocodir, batch_size=32):
    
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    dataloader = create_dataloader(
        f"{cocodir}/train2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=True, hyp=hyp, rect=True, cache=False, stride=32, pad=0, image_weights=False
    )[0]
    return dataloader

# input: Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)

def prepare_model(weight, device):
    # quant_modules.initialize()
    initialize()
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()    # conv bn 进行层的合并, 加速
    return model

def tranfer_torch_to_quantization(nn_instance, quant_module):
    
    quant_instances = quant_module.__new__(quant_module)

    # 属性赋值
    for k, val in vars(nn_instance).items():
        setattr(quant_instances, k, val)

    # 初始化
    def __init__(self):
        # 返回两个 QuantDescriptor 的实例 self.__class__ 是 quant_instance 的类, QuantConv2d
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            # 加快量化速度
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instances)
    return quant_instances

def torch_module_find_quant_module(model, module_list, prefix=''):
    for name in model._modules:
        submodule = model._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_list, prefix=path) # 递归

        submodule_id = id(type(submodule))
        if submodule_id in module_list:
            # 转换
            model._modules[name] = tranfer_torch_to_quantization(submodule, module_list[submodule_id])
        
def replace_to_quantization_model(model):
    
    module_list = {}
    
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)  # module -> torch.nn.modules.conv.Conv1d
        module_list[id(module)] = entry.replace_mod
    
    torch_module_find_quant_module(model, module_list)


def evaluate_coco(model, loader, save_dir='', conf_thres=0.001, iou_thres=0.65):
    
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    return test.test(
        "data/coco.yaml",
        save_dir=Path(save_dir),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        dataloader=loader,
        is_coco=True,
        plots=False,
        half_precision=True,
        save_json=False
    )[0][3]

def collect_stats(model, data_loader, device, num_batch = 200):
    model.eval()

    # 开启校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # test
    with torch.no_grad():
        for i, datas in enumerate(data_loader):
            imgs = datas[0].to(device, non_blocking=True).float() / 255.0
            model(imgs)

            if i >= num_batch:
                break
    
    # 关闭校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
            
def compute_amax(model, **kwargs):
    
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                module._amax = module._amax.to(device)


def calibrate_model(model, dataloader, device):

    # 收集前向信息
    collect_stats(model, dataloader, device)

    # 获取动态范围，计算 amax 值，scale 值
    compute_amax(model, method = 'mse')

if __name__ == "__main__":

    weight = "yolov7.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    print("Evalute Dataset...")
    cocodir = "dataset/coco2017"
    val_dataloader   = prepare_val_dataset(cocodir)
    train_dataloader = prepare_train_dataset(cocodir)

    # 加载 pth 模型
    pth_model = load_yolov7_model(weight, device)
    # pth 模型验证
    print("Evalute Origin...")
    ap = evaluate_coco(pth_model, val_dataloader)

    # 获取伪量化模型(手动 initial(), 手动插入 QDQ)
    model = prepare_model(weight, device)
    replace_to_quantization_model(model)

    # 模型标定
    calibrate_model(model, train_dataloader, device)

    # # PTQ 模型验证
    print("Evaluate PTQ...")
    ptq_ap = evaluate_coco(model, val_dataloader)
```

值得注意的是我们校准时是在训练集上完成的，测试时是在验证集上完成的，运行效果如下：

![0175313e5448829028e4b202628242e5](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(二)/0175313e5448829028e4b202628242e5.png)

可以看到量化校准后的模型的 mAP 仅仅下降了 0.003 个点。

博主学得有点混淆了，先梳理下一些概念，我们收集统计信息的目的是为了确定当前 tensor 的 amax 即幅度的最大值，然后根据不同的校准方法和获取的统计信息去校准计算 amax，其中包括 Max 和直方图两种校准方法，Max 校准方法直接选择 tensor 统计信息的最大值来作为 amax，而直方图校准中又包含 entropy、mse、percentile 三种方法来计算 amax，~上述过程仅仅是进行了校准确定了 amax 值，得到了量化时所需要的 scale，但是还没有利用 scale 进行具体的量化操作，模型的权重或激活值还没有改变，应该是这么理解的吧😂~

**上述过程中进行了校准确定了 amax 值，得到了量化时所需要的 scale，并在模型 forward 的过程中内部执行了量化操作，因此上述流程是进行了 PTQ 量化的**

* * *

* * *

**2023/12/27 新增内容**

博主之前一直以为 Q/DQ 节点是 QAT 量化专属的，这还是属于量化的一些基础概念都没有理清楚😂

实际上 Q/DQ 节点既用于 QAT 量化也用于 PTQ 量化，这两种量化策略的主要区别在于它们使用 Q/DQ 节点的方式和量化的时间点，具体如下：(**from ChatGPT**)

**PTQ 中的 Q/DQ 节点**

*   在 PTQ 量化过程中，Q/DQ 节点被插入到已经训练好的模型中。这是为了模拟量化过程中对模型推理的影响，并通过校准数据来确定最佳的量化参数（如 scale 和 zero-point）
*   在 PTQ 量化过程中，Q/DQ 节点**主要用于量化转换过程中的数据收集和量化参数的确定，它们不参与模型训练的反向传播过程**

**QAT 中的 Q/DQ 节点**

*   在 QAT 量化过程中，Q/DQ 节点是模型训练过程的一部分。它们被用来模拟量化的影响，并在训练过程中调整模型的权重，以最小化量化带来的性能损失
*   在 QAT 量化过程中，Q/DQ 节点**对模型权重的更新有直接影响**。这是因为它们参与了整个训练过程，包括前向传播和反向传播。

所以说 Q/DQ 在 PTQ 和 QAT 中扮演着不同的角色，在 PTQ 中是模拟量化过程确定 scale，而在 QAT 中不仅仅会模拟量化确定 scale 还会在微调训练过程中调整模型的权重以适应量化带来的影响

以下是 QAT 中 Q/DQ 节点作用的详细解释：(**from ChatGPT**)

*   **模拟训练环境**：Q/DQ 节点被引入到巡礼过程中，模拟量化后模型的运行环境。这意味着在训练过程中，权重和激活数据会经历实际的量化和反量化过程。
*   **权重调整**：由于量化过程可能引入一定的误差，在训练过程中，模型会通过标准的梯度下降和反向传播过程，**不断调整权重**。这个过程旨在**使模型适应量化带来的影响，从而减少量化误差对模型性能的影响**
*   **学习量化参数**：同时，QAT 过程中还会学习确定量化过程中的关键参数，如 scale 和 zero-point。这些参数是量化过程中非常关键的，它们决定了如何讲浮点数值映射到整数表示
*   **最终结果**：通过这种方式，QAT 量化后的模型不仅仅是获得了适合量化的 scale 值，而且其权重也被调整为更适合量化后的运行环境，这有助于保持或接近原始浮点模型的性能

QAT 和 PTQ 量化最显著的区别在于 QAT 量化中模型的权重会发生变化以适应量化带来的影响。

简单总结下，PTQ 和 QAT 模型都会携带 Q/DQ 节点，QAT 量化会通过训练的方式获取 scale 等量化信息并调整模型权重以适应量化带来的影响，PTQ 量化则是通过校准图片来获取 scale 等量化信息无需训练

最后再来梳理下二者的区别：(**from ChatGPT**)

**PTQ**

*   **操作时间**：PTQ 是在模型训练完成后进行的。这种方法不涉及重新训练模型
*   **主要步骤**：
    *   **插入 Q/DQ 节点**：首先在模型的适当位置插入量化（Quantize）和反量化（Dequantize）节点
    *   **校准**：通过使用一组代表性数据（通常叫校准数据集）来运行模型，以此来收集激活（Activation）的统计数据。这些数据用于确定量化参数（如 scale 和 zero-point）
    *   **量化转换**：利用收集到的统计数据，将浮点权重和激活转换为整数格式
*   **优势**：操作简单，不需要额外训练，适用于资源有限的情况
*   **劣势**：可能会有较大的精度损失，尤其是对于那些对量化敏感的模型（需要进行敏感层分析）

**QAT**

*   **操作时间**：QAT 是在模型训练过程中进行的。它实际上是模型训练的一个部分。
*   **主要步骤**：
    *   **模拟量化**：在训练过程中引入 Q/DQ 节点，模拟量化过程中的影响。这意味着在前向传播和反向传播时，权重和激活都会经历量化和反量化的过程
    *   **训练微调**：通过对模型的正常训练流程进行微调，调整权重，以补偿量化过程可能引入的误差
    *   **学习量化参数**：在训练过程中学习确定最佳的量化参数（如 scale）
*   **优势**：由于模型在训练过程中已经适应了量化的影响，因此量化后的模型通常有更好的性能和较小的精度损失
*   **劣势**：需要额外的训练资源和时间，相对于 PTQ 来说更加复杂

OK，以上就是本次更新新增的内容，如有不对的地方，欢迎各位看官批评指正😄

* * *

* * *

下面我们来对比下 Max 和直方图校准方法的 PTQ 模型的对比，来看看不同的校准方法对模型的影响

上面我们测试了直方图校准后的 PTQ [模型性能](https://edu.csdn.net/cloud/pm_summit?utm_source=blogglc&spm=1001.2101.3001.7020)，下面我们来看 Max 校准方法，我们将 prepare\_model 函数中的手动 initialize 函数注释，打开自动初始化 quant\_module.initialize

再次执行代码如下所示：

![0a1db5742c80d3fbfc571ceaa95b816b](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(二)/0a1db5742c80d3fbfc571ceaa95b816b.png)

可以看到我们使用默认的 Max 校准方法得到的 mAP 值是 0.444，相比于之前直方图校准的效果要差一些，因此后续我们可能就使用直方图校准的方式来进行量化。

下面我们来看看 PTQ 模型的导出，导出函数如下：

```python
def export_ptq(model, save_file, device, dynamic_batch = True):
    
    input_dummy = torch.randn(1, 3, 640, 640, device=device)
    
    # 打开 fake 算子
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, input_dummy, save_file, opset_version=13,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None)
```

执行后效果如下：

![9fded8c52fb14152c5445db3ce933bb2](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(二)/9fded8c52fb14152c5445db3ce933bb2.png)

我们将导出的 PTQ 模型和原始的 YOLOv7 模型对比，

![23a9074455dd319a5c73c8bc4ce233d6](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(二)/23a9074455dd319a5c73c8bc4ce233d6.png)

左边是我们原始的 ONNX，右边是我们 PTQ 模型的 ONNX，可以看到导出的 PTQ 模型中多了 QDQ 节点的插入，其中包含了校准量化信息 scale。

以上就是 torch 和 PTQ 模型的对比，下面我们来进行敏感层的分析。

# 3\. 敏感层分析

我们先梳理下敏感层分析的流程：

*   **1.** for 循环 model 的每一个 quantizer 层
*   **2.** 只关闭该层的量化，其余层的量化保留
*   **3.** 验证模型的精度，evaluate\_coco(), 并保存精度值
*   **4.** 验证结束，重启该层的量化操作
*   **5.** for 循环结束，得到所有层的精度值
*   **6.** 排序，得到前 10 个对精度影响比较大的层，将这些层进行打印输出

类似于控制变量法，关闭某一层的量化看精度下降幅度，选出对精度影响最大的几个层作为敏感层。

我们来按照上述流程编写代码即可，首先是 **sensitive\_analysis** 函数的实现，代码如下：

```python
def sensitive_analysis(model, loader):
    
    save_file = "senstive_analysis.json"

    summary =  SummaryTools(save_file)

    # for 循环每一个层
    print(f"Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断 layer 是否是量化层
        if have_quantizer(layer):   # 如果是量化层
            # 使该层的量化失效，不进行 int8 的量化，使用 fp16 精度运算
            disable_quantization(layer).apply()

            # 计算 map 值
            ap = evaluate_coco(model, loader )

            # 保存精度值，json 文件
            summary.append([ap, f"model.{i}"])
            print(f"layer {i} ap: {ap}")

            # 重启层的量化，还原
            enable_quantization(layer).apply()
            
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    # 循环结束，打印前 10 个影响比较大的层
    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive Summary")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")
```

该函数是敏感层分析的主要函数，其具体实现流程如下：

*   循环遍历模型的每一层，通过使用 **have\_quantizer** 函数来检查层是否为量化层
*   使用 **disable\_quantization** 和 **enable\_quantization** 类来关闭和重启量化
*   使用之前的 **evaluate\_coco** 函数来计算 mAP 值
*   使用 **SummaryTools** 类来保存每层的评估结果
*   最后打印前 10 个对精度影响最大的层

下面我们来看看其中调用的函数和类的具体实现

首先是 **have\_quantizer** 函数，其具体实现如下：

```python
# 判断层是否是量化层
def have_quantizer(layer):
    for name, module in layer.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True

    return False
```

该函数的功能是检查传入的层是否为量化层，通过遍历该层的所有模块，检测是否有 **quant\_nn.TensorQuantizer** 的模块，如果有则返回 True，代表该层为量化层，否则返回 False。

然后是 **disable\_quantization** 和 **enable\_quantization** 类，其具体实现如下：

```python
class disable_quantization:

    # 初始化
    def __init__(self, model):
        self.model = model

    # 应用 关闭量化
    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(disabled=True)
    
    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)

# 重启量化
class enable_quantization:
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
            
    def __enter__(self):
        self.apply(enabled=True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False) 
```

它们的功能是分别用于临时关闭和重启模型中的量化操作。这两个类在构造时会接收模型对象，并在 **apply** 方法中遍历模型的所有模块，根据量化状态（启用/禁用）设置 **module.\_disabled** 属性。

最后是 **SummaryTools** 类，其实现如下：

```python
import json
class SummaryTools:

    def __init__(self, file):
        self.file = file
        self.data = []
    
    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)
```

该类的功能是用于保存每层的 mAP 结果。在其 **append** 方法中会添加 mAP 结果到内部数据列表，并将这些数据保存到 JSON 文件中。

完整的敏感层分析代码如下：

```python
import os
import yaml
import test
import torch
import collections
from pathlib import Path
from models.yolo import Model
from pytorch_quantization import calib
from absl import logging as quant_logging
from utils.datasets import create_dataloader
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules import _utils as quant_nn_utils

def load_yolov7_model(weight, device='cpu'):
    ckpt  = torch.load(weight, map_location=device)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model

def prepare_val_dataset(cocodir, batch_size=32):
    dataloader = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader

def prepare_train_dataset(cocodir, batch_size=32):
    
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    dataloader = create_dataloader(
        f"{cocodir}/train2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=True, hyp=hyp, rect=True, cache=False, stride=32, pad=0, image_weights=False
    )[0]
    return dataloader

# input: Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)

def prepare_model(weight, device):
    # quant_modules.initialize()
    initialize()
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()    # conv bn 进行层的合并, 加速
    return model

def tranfer_torch_to_quantization(nn_instance, quant_module):
    
    quant_instances = quant_module.__new__(quant_module)

    # 属性赋值
    for k, val in vars(nn_instance).items():
        setattr(quant_instances, k, val)

    # 初始化
    def __init__(self):
        # 返回两个 QuantDescriptor 的实例 self.__class__ 是 quant_instance 的类, QuantConv2d
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            # 加快量化速度
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instances)
    return quant_instances

import re
def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False
    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer]
        if path in ignore_layer:
            return True
        for item in ignore_layer:
            if re.match(item, path):
                return True
    return False

def torch_module_find_quant_module(model, module_list, ignore_layer, prefix=''):
    for name in model._modules:
        submodule = model._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_list, ignore_layer, prefix=path) # 递归

        submodule_id = id(type(submodule))
        if submodule_id in module_list:
            ignored = quantization_ignore_match(ignore_layer, path)
            if ignored:
                print(f"Quantization : {path} has ignored.")
                continue
            # 转换
            model._modules[name] = tranfer_torch_to_quantization(submodule, module_list[submodule_id])
        
def replace_to_quantization_model(model, ignore_layer=None):
    
    module_list = {}
    
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)  # module -> torch.nn.modules.conv.Conv1d
        module_list[id(module)] = entry.replace_mod
    
    torch_module_find_quant_module(model, module_list, ignore_layer)


def evaluate_coco(model, loader, save_dir='', conf_thres=0.001, iou_thres=0.65):
    
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    return test.test(
        "data/coco.yaml",
        save_dir=Path(save_dir),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        dataloader=loader,
        is_coco=True,
        plots=False,
        half_precision=True,
        save_json=False
    )[0][3]

def collect_stats(model, data_loader, device, num_batch = 200):
    model.eval()

    # 开启校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # test
    with torch.no_grad():
        for i, datas in enumerate(data_loader):
            imgs = datas[0].to(device, non_blocking=True).float() / 255.0
            model(imgs)

            if i >= num_batch:
                break
    
    # 关闭校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
            
def compute_amax(model, **kwargs):
    
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                module._amax = module._amax.to(device)


def calibrate_model(model, dataloader, device):

    # 收集前向信息
    collect_stats(model, dataloader, device)

    # 获取动态范围，计算 amax 值，scale 值
    compute_amax(model, method = 'mse')

def export_ptq(model, save_file, device, dynamic_batch = True):
    
    input_dummy = torch.randn(1, 3, 640, 640, device=device)
    
    # 打开 fake 算子
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, input_dummy, save_file, opset_version=13,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None)

    quant_nn.TensorQuantizer.use_fb_fake_quant = False

# 判断层是否是量化层
def have_quantizer(layer):
    for name, module in layer.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True

    return False

class disable_quantization:

    # 初始化
    def __init__(self, model):
        self.model = model

    # 应用 关闭量化
    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(disabled=True)
    
    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)

# 重启量化
class enable_quantization:
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
            
    def __enter__(self):
        self.apply(enabled=True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)    

import json
class SummaryTools:

    def __init__(self, file):
        self.file = file
        self.data = []
    
    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


def sensitive_analysis(model, loader):
    
    save_file = "senstive_analysis.json"

    summary =  SummaryTools(save_file)

    # for 循环每一个层
    print(f"Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断 layer 是否是量化层
        if have_quantizer(layer):   # 如果是量化层
            # 使该层的量化失效，不进行 int8 的量化，使用 fp16 精度运算
            disable_quantization(layer).apply()

            # 计算 map 值
            ap = evaluate_coco(model, loader )

            # 保存精度值，json 文件
            summary.append([ap, f"model.{i}"])
            print(f"layer {i} ap: {ap}")

            # 重启层的量化，还原
            enable_quantization(layer).apply()
            
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    # 循环结束，打印前 10 个影响比较大的层
    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive Summary")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")


if __name__ == "__main__":

    weight = "yolov7.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    print("Evalute Dataset...")
    cocodir = "dataset/coco2017"
    val_dataloader   = prepare_val_dataset(cocodir)
    train_dataloader = prepare_train_dataset(cocodir)

    # 加载 pth 模型
    # pth_model = load_yolov7_model(weight, device)
    # pth 模型验证
    # print("Evalute Origin...")
    # ap = evaluate_coco(pth_model, val_dataloader)

    # 获取伪量化模型(手动 initial(), 手动插入 QDQ)
    model = prepare_model(weight, device)
    replace_to_quantization_model(model)

    # 模型标定
    calibrate_model(model, train_dataloader, device)

    # 敏感层分析
    """
    流程:
    1. for 循环 model 的每一个 quantizer 层
    2. 只关闭该层的量化，其余层的量化保留
    3. 验证模型的精度, evaluate_coco(), 并保存精度值
    4. 验证结束，重启该层的量化操作
    5. for 循环结束，得到所有层的精度值
    6. 排序，得到前 10 个对精度影响比较大的层，将这些层进行打印输出
    """
    sensitive_analysis(model, val_dataloader)
    
    # PTQ 模型验证
    # print("Evaluate PTQ...")
    # ptq_ap = evaluate_coco(model, val_dataloader)

    # PTQ 模型导出
    # print("Export PTQ...")

    # export_ptq(model, "ptq_yolov7.onnx", device)
```

在代码中我们关闭了某些不必要的操作，执行后运行效果如下：

![9b42b85a0aa80b64c6f0071e7277a486](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(二)/9b42b85a0aa80b64c6f0071e7277a486.png)

从上图中可以看出它会计算每层关闭量化后的 mAP 值，每层的 mAP 值都不一样，这说明不同层量化对最终精度影响的效果不同，最后我们会将每层的 mAP 值都保存并统计前 10 个对精度影响最大的层。

敏感层的分析等待时间会比较久，因为每层都要去计算 mAP 值。由于博主[硬件](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782?utm_source=glcblog&spm=1001.2101.3001.7020)的原因，没有跑完所有层的分析，后续是直接选用视频中的 10 个层作为敏感层。

视频中分析出来的前 10 个敏感层如下：

```python
ignore_layer = ["model\.104\.(.*)", "model\.37\.(.*)", "model\.2\.(.*)", "model\.1\.(.*)", "model\.77\.(.*)",
                "model\.99\.(.*)", "model\.70\.(.*)", "model\.95\.(.*)", "model\.92\.(.*)", "model\.81\.(.*)"]
```

OK！上面我们对敏感层进行了一个分析，并且将前 10 个对精度影响最大的层进行了打印，接下来我们将处理敏感层分析出来的结果，对精度影响较大的层关闭它的量化，使用 FP16 进行计算

我们在进行 PTQ 量化前就要进行敏感层的分析，得到影响比较大的层，然后在使用手动插入 QDQ 量化节点的时候将这些敏感层传递进来，将其量化进行关闭，这就是我们对敏感层的处理。

因此我们在之前的 **replace\_to\_quantization\_model** 函数中需要多传入一个参数，即上面的敏感层列表，修改后的函数具体实现如下：

```python
def replace_to_quantization_model(model, ignore_layer=None):
    
    module_list = {}
    
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)  # module -> torch.nn.modules.conv.Conv1d
        module_list[id(module)] = entry.replace_mod
    
    torch_module_find_quant_module(model, module_list, ignore_layer)
```

接着我们会将 **ignore\_layer** 列表传入到 **torch\_module\_find\_quant\_module** 函数中，在量化转换时忽略这些层，修改后的函数具体实现如下：

```python
def torch_module_find_quant_module(model, module_list, ignore_layer, prefix=''):
    for name in model._modules:
        submodule = model._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_list, ignore_layer, prefix=path) # 递归

        submodule_id = id(type(submodule))
        if submodule_id in module_list:
            ignored = quantization_ignore_match(ignore_layer, path)
            if ignored:
                print(f"Quantization : {path} has ignored.")
                continue
            # 转换
            model._modules[name] = tranfer_torch_to_quantization(submodule, module_list[submodule_id])
```

该函数功能还是遍历模型的每个子模块，检查是否应该进行量化转换。但与之前不同的是我们新增了一个判断，我们会使用 **quantization\_ignore\_match** 函数来判断当前子模块是否在 **ignore\_layer** 列表中，如果在则跳过量化转换开始下一个模块，如果不在则执行量化转换。

**quantization\_ignore\_match** 的具体实现如下：

```python
import re
def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False
    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer]
        if path in ignore_layer:
            return True
        for item in ignore_layer:
            if re.match(item, path):
                return True
    return False
```

该函数的功能是判断模型中的某一个层是否在 **ignore\_layer** 列表中，即是否应该忽略该层的量化，返回值是一个布尔值。**ignore\_layer** 可以是字符串或列表，我们将使用正则表达式 **re.match** 来检查 **path** 是否能和 **ignore\_layer** 列表中的元素匹配上。

我们将上述代码修改好后，再来测试下，看忽略这些层后量化节点的插入是否发生变化，测试的运行效果如下：

![ba68500dd729d1e17a0d5f6f5e5cd043](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(二)/ba68500dd729d1e17a0d5f6f5e5cd043.png)

可以看到我们打印了忽略某些层的量化后插入 QDQ 节点的模型结构，我们从图中可以看到 99 层是我们忽略的层，它并没有 \_input\_quantizer 和 \_weight\_quantizer，说明它并没有被插入量化节点，使用的是 FP16 的计算，同理 104 层也是如此。

那以上就是敏感层的分析，以及我们根据敏感层的结果对敏感层的量化进行关闭的内容了。

下面我们再来梳理下 PTQ 量化

## 4\. PTQ量化

这节我们将 PTQ 的代码进行工程化

首先编写一个 **quantize.py** 将我们之前的编写的函数和类放入其中，其具体内容如下：

```python
import os
import yaml
import test
import json
import torch
import collections
from pathlib import Path
from models.yolo import Model
from pytorch_quantization import calib
from absl import logging as quant_logging
from utils.datasets import create_dataloader
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules import _utils as quant_nn_utils

def load_yolov7_model(weight, device='cpu'):
    ckpt  = torch.load(weight, map_location=device)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model

def prepare_val_dataset(cocodir, batch_size=32):
    dataloader = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader

def prepare_train_dataset(cocodir, batch_size=32):
    
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    dataloader = create_dataloader(
        f"{cocodir}/train2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=True, hyp=hyp, rect=True, cache=False, stride=32, pad=0, image_weights=False
    )[0]
    return dataloader

# input: Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)

def prepare_model(weight, device):
    # quant_modules.initialize()
    initialize()
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()    # conv bn 进行层的合并, 加速
    return model

def tranfer_torch_to_quantization(nn_instance, quant_module):
    
    quant_instances = quant_module.__new__(quant_module)

    # 属性赋值
    for k, val in vars(nn_instance).items():
        setattr(quant_instances, k, val)

    # 初始化
    def __init__(self):
        # 返回两个 QuantDescriptor 的实例 self.__class__ 是 quant_instance 的类, QuantConv2d
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            # 加快量化速度
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instances)
    return quant_instances

import re
def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False
    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer]
        if path in ignore_layer:
            return True
        for item in ignore_layer:
            if re.match(item, path):
                return True
    return False

def torch_module_find_quant_module(model, module_list, ignore_layer, prefix=''):
    for name in model._modules:
        submodule = model._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_list, ignore_layer, prefix=path) # 递归

        submodule_id = id(type(submodule))
        if submodule_id in module_list:
            ignored = quantization_ignore_match(ignore_layer, path)
            if ignored:
                print(f"Quantization : {path} has ignored.")
                continue
            # 转换
            model._modules[name] = tranfer_torch_to_quantization(submodule, module_list[submodule_id])
        
def replace_to_quantization_model(model, ignore_layer=None):
    
    module_list = {}
    
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)  # module -> torch.nn.modules.conv.Conv1d
        module_list[id(module)] = entry.replace_mod
    
    torch_module_find_quant_module(model, module_list, ignore_layer)


def evaluate_coco(model, loader, save_dir='', conf_thres=0.001, iou_thres=0.65):
    
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    return test.test(
        "data/coco.yaml",
        save_dir=Path(save_dir),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        dataloader=loader,
        is_coco=True,
        plots=False,
        half_precision=True,
        save_json=False
    )[0][3]

def collect_stats(model, data_loader, device, num_batch = 200):
    model.eval()

    # 开启校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # test
    with torch.no_grad():
        for i, datas in enumerate(data_loader):
            imgs = datas[0].to(device, non_blocking=True).float() / 255.0
            model(imgs)

            if i >= num_batch:
                break
    
    # 关闭校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
            
def compute_amax(model, device, **kwargs):

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                module._amax = module._amax.to(device)


def calibrate_model(model, dataloader, device):

    # 收集前向信息
    collect_stats(model, dataloader, device)

    # 获取动态范围，计算 amax 值，scale 值
    compute_amax(model, device, method = 'mse')

def export_ptq(model, save_file, device, dynamic_batch = True):
    
    input_dummy = torch.randn(1, 3, 640, 640, device=device)
    
    # 打开 fake 算子
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, input_dummy, save_file, opset_version=13,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None)

    quant_nn.TensorQuantizer.use_fb_fake_quant = False

# 判断层是否是量化层
def have_quantizer(layer):
    for name, module in layer.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True

    return False

class disable_quantization:

    # 初始化
    def __init__(self, model):
        self.model = model

    # 应用 关闭量化
    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(disabled=True)
    
    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)

# 重启量化
class enable_quantization:
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
            
    def __enter__(self):
        self.apply(enabled=True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)    

class SummaryTools:

    def __init__(self, file):
        self.file = file
        self.data = []
    
    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


def sensitive_analysis(model, loader):
    
    save_file = "senstive_analysis.json"

    summary =  SummaryTools(save_file)

    # for 循环每一个层
    print(f"Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断 layer 是否是量化层
        if have_quantizer(layer):   # 如果是量化层
            # 使该层的量化失效，不进行 int8 的量化，使用 fp16 精度运算
            disable_quantization(layer).apply()

            # 计算 map 值
            ap = evaluate_coco(model, loader )

            # 保存精度值，json 文件
            summary.append([ap, f"model.{i}"])
            print(f"layer {i} ap: {ap}")

            # 重启层的量化，还原
            enable_quantization(layer).apply()
            
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    # 循环结束，打印前 10 个影响比较大的层
    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive Summary")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")
```

这就是我们之前用于 YOLOv7-PTQ 量化的各种函数和类的实现，这里不再赘述

另外我们新建一个 **ptq.py** 文件，用于实现 YOLOv7 的 PTQ 量化，我们通过 **argparse** 模块来传入 PTQ 量化所需要的参数，代码如下：

```python
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cocodir', type=str,  default="dataset/coco2017", help="coco directory")
    parser.add_argument('--batch_size', type=int,  default=8, help="batch size for data loader")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--sensitive', type=bool, default=True, help="use sensitive analysis or not befor ptq")
    parser.add_argument("--sensitive_summary", type=str, default="sensitive-summary.json", help="summary save file")
    parser.add_argument("--ignore_layers", type=str, default="model\.105\.m\.(.*)", help="regx")
    
    parser.add_argument("--save_ptq", type=bool, default=False, help="file")
    parser.add_argument("--ptq", type=str, default="ptq_yolov7.onnx", help="file")
    
    parser.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--nmsthres", type=float, default=0.65, help="nms threshold")
    
    parser.add_argument("--eval_origin", action="store_true", help="do eval for origin model")
    parser.add_argument("--eval_ptq", action="store_true", help="do eval for ptq model")
    
    parser.add_argument("--ptq_summary", type=str, default="ptq_summary.json", help="summary save file")
    
    args = parser.parse_args()
```

传入的参数有权重、数据集路径的指定，敏感层分析的指定，置信度阈值的指定等等

我们可以通过调用 **quantize.py** 模块的各种函数和类来实现真正的量化，量化主要分为敏感层分析和 PTQ 量化两个部分，我们可以分别编写两个函数来调用实现，首先是敏感层分析函数，其实现如下：

```python
def run_SensitiveAnalysis(weight, cocodir, device='cpu'):

    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(weight, device)
    quantize.replace_to_quantization_model(model)

    # prepare dataset
    print("Prepare Dataset ....")
    train_dataloader = quantize.prepare_train_dataset(cocodir)
    val_dataloader = quantize.prepare_val_dataset(cocodir)

    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)

    # sensitive analysis
    print("Begining Sensitive Analysis ....")
    quantize.sensitive_analysis(model, val_dataloader, args.sensitive_summary)
```

我们在前面就讲过敏感层分析的流程，包括模型、数据集的准备、模型的标定，敏感层的分析，都是通过 **quantize.py** 模块的各种函数和类来实现的

我们再来编写下运行 PTQ 量化的函数，其实现如下：

```python
def run_PTQ(args, device='cpu'):

    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(args.weights, device)
    quantize.replace_to_quantization_model(model, args.ignore_layers)

    # prepare dataset
    print("Prepare Dataset ....")
    val_dataloader = quantize.prepare_val_dataset(args.cocodir, batch_size=args.batch_size)
    train_dataloader = quantize.prepare_train_dataset(args.cocodir, batch_size=args.batch_size)
    
    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)
    
    summary = quantize.SummaryTool(args.ptq_summary)
    
    if args.eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
            summary.append(["Origin", ap])
    if args.eval_ptq:
        print("Evaluate PTQ...")
        ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
        summary.append(["PTQ", ap])

    if args.save_ptq:
        print("Export PTQ...")
        quantize.export_ptq(model, args.ptq, device)
```

实际的 PTQ 量化过程包括权重、数据集的准备，标定，后续 PTQ 模型性能的验证和导出

那以上就是 **ptq.py** 文件中的全部内容，完整的内容如下：

```python
import torch
import quantize
import argparse

def run_SensitiveAnalysis(weight, cocodir, device='cpu'):

    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(weight, device)
    quantize.replace_to_quantization_model(model)

    # prepare dataset
    print("Prepare Dataset ....")
    train_dataloader = quantize.prepare_train_dataset(cocodir)
    val_dataloader = quantize.prepare_val_dataset(cocodir)

    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)

    # sensitive analysis
    print("Begining Sensitive Analysis ....")
    quantize.sensitive_analysis(model, val_dataloader, args.sensitive_summary)

def run_PTQ(args, device='cpu'):

    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(args.weights, device)
    quantize.replace_to_quantization_model(model, args.ignore_layers)

    # prepare dataset
    print("Prepare Dataset ....")
    val_dataloader = quantize.prepare_val_dataset(args.cocodir, batch_size=args.batch_size)
    train_dataloader = quantize.prepare_train_dataset(args.cocodir, batch_size=args.batch_size)
    
    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)
    
    summary = quantize.SummaryTool(args.ptq_summary)
    
    if args.eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
            summary.append(["Origin", ap])
    if args.eval_ptq:
        print("Evaluate PTQ...")
        ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
        summary.append(["PTQ", ap])

    if args.save_ptq:
        print("Export PTQ...")
        quantize.export_ptq(model, args.ptq, device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cocodir', type=str,  default="dataset/coco2017", help="coco directory")
    parser.add_argument('--batch_size', type=int,  default=8, help="batch size for data loader")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--sensitive', type=bool, default=True, help="use sensitive analysis or not befor ptq")
    parser.add_argument("--sensitive_summary", type=str, default="sensitive-summary.json", help="summary save file")
    parser.add_argument("--ignore_layers", type=str, default="model\.105\.m\.(.*)", help="regx")
    
    parser.add_argument("--save_ptq", type=bool, default=False, help="file")
    parser.add_argument("--ptq", type=str, default="ptq_yolov7.onnx", help="file")
    
    parser.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--nmsthres", type=float, default=0.65, help="nms threshold")
    
    parser.add_argument("--eval_origin", action="store_true", help="do eval for origin model")
    parser.add_argument("--eval_ptq", action="store_true", help="do eval for ptq model")
    
    parser.add_argument("--ptq_summary", type=str, default="ptq_summary.json", help="summary save file")
    
    args = parser.parse_args()

    is_cuda = (args.device != "cpu") and torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else 'cpu')

    # 敏感层分析
    if args.sensitive:
        print("Sensitive Analysis ...")
        run_SensitiveAnalysis(args.weights, args.cocodir, device)

    # PTQ
    # ignore_layers= ["model\.105\.m\.(.*)", model\.99\.m\.(.*)]
    # args.ignore_layer = ignore_layers
    
    print("Begining PTQ ....")
    run_PTQ(args, device)
    print("PTQ Quantization Has Finished ....")
```

那其实这都是我们之前讲过的内容，只是这边再重新整理并工程化下，方便我们后续的使用。

OK！YOLOv7-PTQ 量化的内容到这里就结束了，下节开始我们将讲解 QAT 量化相关的知识

# 总结

> 本次课程介绍了 YOLOv7-PTQ 量化流程中的标定、敏感层分析，标定主要是利用标定数据来收集模型中各层的统计信息，并计算量化参数保存在 QDQ 节点当中，此外我们还对比了 Max 和 直方图校准两种方法，发现 Max 方法的性能要差一些，而敏感层分析的流程则是循环遍历所有层，关闭某层量化测试 mAP 性能，最终统计对模型性能最大的几个层作为敏感层，关闭其量化以 FP16 的方式运行，那我们在实际进行 PTQ 量化之前就要做敏感层的分析，统计出哪些层是敏感层后再进行量化，这样量化出的模型的性能也会更高。最后 PTQ 量化模型的导出记得打开 fake 算子，也就是将 use\_fb\_fake\_quant 设置为 True。
> 
> 至此，YOLOv7-PTQ 量化的全部内容到这里就讲完了，下节开始我们将进入 YOLOv7-QAT 量化

本文转自 <https://blog.csdn.net/qq_40672115/article/details/134233620>，如有侵权，请联系删除。