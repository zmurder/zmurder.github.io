 

# 目录

*   *   [注意事项](#_2)
    *   [一、2023/11/5更新](#2023115_3)
    *   [前言](#_6)
    *   [1\. YOLOv7-PTQ量化流程](#1_YOLOv7PTQ_19)
    *   [2\. 准备工作](#2__69)
    *   [3\. 插入QDQ节点](#3_QDQ_306)
    *   *   [3.1 自动插入QDQ节点](#31_QDQ_308)
        *   [3.2 手动插入QDQ节点](#32_QDQ_630)
        *   [3.3 手动initialize](#33_initialize_833)
    *   [总结](#_872)

# 注意事项

### 一、2023/11/5更新

**新增手动插入 QDQ 节点以及手动 initialize**

### 前言

> 手写 AI 推出的全新 TensorRT [模型](https://ml-summit.org/cloud-member?uid=c1041&spm=1001.2101.3001.7020)量化实战课程，[链接](https://www.bilibili.com/video/BV1NN411b7HZ/?spm_id_from=333.999.0.0)。记录下个人学习笔记，仅供自己参考。
> 
> 该实战课程主要基于手写 AI 的 Latte 老师所出的 [TensorRT下的模型量化](https://www.bilibili.com/video/BV18L41197Uz/)，在其课程的基础上，所整理出的一些实战应用。
> 
> 本次课程为 YOLOv7 量化实战第二课，主要介绍 YOLOv7-PTQ 量化
> 
> 课程大纲可看下面的思维导图

![3ac8e54d0077179cfb351dafc2574475](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/3ac8e54d0077179cfb351dafc2574475.png)

# 1\. YOLOv7-PTQ量化流程

> 在上节课程中我们介绍了 pytorch\_quantization 量化工具箱，从这节课开始我们将正式进入 YOLOv7-PTQ 量化的实战。
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

**6.** **性能对比**

第六个就是[性能](https://marketing.csdn.net/p/3127db09a98e0723b83b2914d9256174?pId=2782?utm_source=glcblog&spm=1001.2101.3001.7020)的对比，包括精度和速度的对比。

OK！以上就是 YOLOv7-PTQ 量化的流程，下面我们根据上面的流程来具体的实现，让我们开始吧！！！🚀🚀🚀

# 2\. 准备工作

首先是我们的准备工作，在正式开始前我们需要准备三个东西：

*   **代码**：yolov7 官方代码
*   **数据集**：coco2017
*   **官方预训练模型**：yolov7.pt

大家可以点击 [here【pwd:yolo】](https://pan.baidu.com/s/1hXBxi9Sm_iW6nw5BFYxy-g) 下载博主准备好的相关代码、模型和数据集

我们来看下我们整个项目的目录结构，如下图所示：

![58b081510492e028789108b8fcb87d42](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/58b081510492e028789108b8fcb87d42.png)

其中的 coco2017 的数据集目录如下：

```shell
.
├─train2017
│  ├─images
│  ├─labels
│  └─xml
└─val2017
    ├─images
    ├─labels
    └─xml
```

除此之外我们还需要 train2017.txt 和 val2017.txt 两个 TXT 文件，分别存储着对应[训练集](https://ml-summit.org/cloud-member?uid=c1041&spm=1001.2101.3001.7020)和验证集**图像的完整路径**，以下是生成对应 TXT 的代码：

```python
import os

save_dir  = "/home/jarvis/Learn/Datasets/VOC_QAT"
train_dir = "/home/jarvis/Learn/Datasets/VOC_QAT/images/train"
train_txt_path = os.path.join(save_dir, "train2017.txt")

with open(train_txt_path, "w") as f:
    for filename in os.listdir(train_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"): # 添加你的图像文件扩展名
            file_path = os.path.join(train_dir, filename)
            f.write(file_path + "\n")

print(f"train2017.txt has been created at {train_txt_path}")

val_dir = "/home/jarvis/Learn/Datasets/VOC_QAT/images/val"
val_txt_path = os.path.join(save_dir, "val2017.txt")

with open(val_txt_path, "w") as f:
    for filename in os.listdir(val_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"): # 添加你的图像文件扩展名
            file_path = os.path.join(val_dir, filename)
            f.write(file_path + "\n")

print(f"val2017.txt has been created at {val_txt_path}")
```

你需要修改以下几项：

*   **save\_dir**：txt 文档保存的路径，应该与 train2017 和 val2017 文件夹在同一级目录
*   **train\_dir**：训练集图片路径
*   **val\_dir**：验证集图片路径

将上述工作完成后，下面我们正式开始编写代码。

我们将数据集和权重文件都放在 YOLOv7-main 文件夹下，并先新建一个 **ptq.py** 文件，先完成模型和数据集加载以及模型 mAP 测试工作，主要是以下三个函数的编写：

*   **load\_yolov7\_model**：加载 YOLOv7 模型权重
*   **prepare\_dataset**：加载数据
*   **evaluate\_coco()**：mAP 测试

我们先看模型加载函数的编写，代码如下：

```python
def load_yolov7_model(weight, device='cpu'):
    ckpt  = torch.load(weight, map_location=device)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model
```

首先我们通过 torch 加载了预训练权重，然后通过 YOLOv7 官方的 Model 类创建了一个实例，并通过 load\_state\_dict 方法将状态字典加载到模型中，最后返回模型。值得大家注意的是我们会将加载的模型权重转换为单精度浮点数，这是因为我们加载的权重可能是 float64，但是我们模型通常在前向的时候使用的是单精度 float32 进行的推理，所以这边做一个转化。

接着我们来看数据集加载函数的编写，代码如下：

```python
def prepare_dataset(cocodir, batch_size=4):
    dataloader = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader
```

我们使用 YOLOv7 官方提供的数据加载器函数 create\_dataloader 完成数据加载，我们将对应的参数填入即可，其中的 opt 参数是用来指定当前数据集是否为单类别数据集，由于我们使用的是 COCO 数据集，其中包含 80 个类别，我们应该设置为 False。

在代码中我们是使用 python 的 **collections.namedtuple** 函数实例化了一个名为 **Opt** 的命名元组类，它有一个字段 **single\_cls**，其被设置为 **False**，那其实就相当于 **opt.single\_cls = Flase** 参数传递进去了。

最后我们来看验证函数的编写，代码如下：

```python
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
```

我们使用的是 YOLOv7 官方的 test 函数，将对应的参数传递即可。

完整的示例代码如下：

```python
import os
import test
import torch
import collections
from pathlib import Path
from models.yolo import Model
from utils.datasets import create_dataloader

def load_yolov7_model(weight, device='cpu'):
    ckpt  = torch.load(weight, map_location=device)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model

def prepare_dataset(cocodir, batch_size=4):
    dataloader = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader

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

if __name__ == "__main__":

    weight = "yolov7.pt"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_yolov7_model(weight, device)

    cocodir = "dataset/coco2017"
    dataloader = prepare_dataset(cocodir)

    ap = evaluate_coco(model, dataloader)
```

在正式开始测试之前，我们还需要修改下 data/coco.yaml 文件，主要修改以下几点：

*   注释第 4 行的数据下载
*   修改第 7 行和 第 8 行的 txt 路径
*   注释第 9 行的 test 路径

完整的 coco.yaml 文件内容如下：

```yaml
# COCO 2017 dataset http://cocodataset.org

# download command/URL (optional)
# download: bash ./scripts/get_coco.sh

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: D:\\YOLO\\yolov7-qat\\yolov7-main\\dataset\\coco2017\\train2017.txt  # 118287 images
val: D:\\YOLO\\yolov7-qat\\yolov7-main\\dataset\\coco2017\\val2017.txt  # 5000 images
# test: ./coco/test-dev2017.txt  # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# number of classes
nc: 80

# class names
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
```

修改完成之后我们就可以在终端执行如下指令完成 mAP 的测试了，指令如下：

```shell
python ptq.py
```

如下图所示：

![075ea1a990307b383f44c3dd7ea2304e](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/075ea1a990307b383f44c3dd7ea2304e.png)

可以看到我们最终测试的 mAP@.5:.95 值是 0.454，那以上就是我们测试 mAP 的一个简单流程了，大家可以自行测试。

下面我们来看如何在模型中插入 QDQ 节点。

# 3\. 插入QDQ节点

## 3.1 自动插入QDQ节点

我们先来看自动插入 QDQ 节点，编写 **prepare\_model** 函数，代码如下：

```python
def prepare_model(weight, device):
    quant_modules.initialize()
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()    # conv bn 进行层的合并, 加速
    return model
```

我们使用 initialize 函数来自动插入 QDQ 节点，我们打印对比下原来的 torch 模型和插入 QDQ 节点模型结构的变化，如下图所示：

![e130471d29babd681626fb5413ac083f](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/e130471d29babd681626fb5413ac083f.png)

**图3-1 原始torch模型**

![d06c38807b1aaa233e9d24c9cbd5e86a](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/d06c38807b1aaa233e9d24c9cbd5e86a.png)

**图3-2 插入QDQ节点模型**

从上图可以看出 torch 模型的结构是我们常见的一个卷积层的结构，而插入了量化节点的模型结构可以看到多了两个输入 \_input\_quantizer 和 \_weight\_quantizer，另外 Conv2d 也变成了对应的量化版 QuantConv2d。

至此，QDQ 节点的自动插入就完成了。

下面我们来了解下 initializer 具体的工作流程，函数定义如下：

```python
def initialize(float_module_list=None, custom_quant_modules=None):
    """Dynamic module replacement using monkey patching.

    Dynamically monkey patches the modules with their quantized versions. Internally, the
    state is maintained by a helper class object which helps in replacing the original
    modules back.

    Args:
        float_module_list: A list. User supplied list which indicates which modules to not monkey patch.
        custom_quant_modules: A dict. A mapping provided by user to indicate any other module apart
            from torch.nn and its corresponding quantized version.

    Returns:
        nothing.

    Typical usage example:

        # Define the deny list for torch.nn modules and custom map for modules other than torch.nn.
        float_module_list = ["Linear"]
        custom_quant_modules = [(torch.nn, "Linear", quant_nn.QuantLinear)]
        ## Monkey patch the modules
        pytorch_quantization.quant_modules.initialize(float_module_list, custom_modules)
        ## Use the quantized modules
        pytorch_quantization.quant_modules.deactivate()
    """
    _quant_module_helper_object.prepare_state(float_module_list, custom_quant_modules)
    _quant_module_helper_object.apply_quant_modules()
```

首先 **initialize** 函数属于 **pytorch\_quantization.quant\_modules** 模块，它用于初始化量化过程，通过所谓的 **monkey patching** 动态地替换模型中的模块为它们的量化版本。

它包含以下两个参数：

*   **float\_module\_list**：用户提供的列表，用来指示哪些模块不应该被替换为量化版本。这允许用户对哪些模块进行量化有更细粒度的控制
*   **custom\_quant\_modules**：用户提供的字典，可以用于指示除了 **torch.nn** 之外的其他模块及其对应的量化版本。这允许用户为自定义的模块指定量化版本

它的工作流程包含以下两个步骤：

**1.** **准备状态**

*   使用 **prepare\_state** 函数来准备量化的状态
*   这个函数接收 **float\_module\_list** 和 **custom\_quant\_modules** 参数，并将这些信息传递给一个辅助类对象
*   辅助类对象使用这些信息来确定哪些模块应该被替换为量化版本，哪些应该保持原样

**2.** **应用量化模块**

*   接下来，**apply\_quant\_modules** 函数来实际应用量化
*   在这一步中，原始的模块被它们的量化版本所替换。对于 **torch.nn** 中的标准模块，比如说 **torch.nn.Conv2d** 会被替换为 **quant\_nn.QuantConv2d**
*   对于用户指定的自定义模块，将使用 **custom\_quant\_modules** 中提供的映射来进行替换

我们再来看下具体的实现模块替换的 QuantModuleReplacementHelper 类，它的结构和功能如下：

**类属性**

*   **orginal\_func\_map**：用于存储原始模块，这些原始模块在进行 **monkey patching** 时会被替换
*   **default\_quant\_map**：存储 pytorch\_quantization 工具所支持的默认量化模块的列表，这些是内置的量化版本，通常对应于 torch.nn 中的标准模块，比如 Conv、Pool、LSTM 等，值得注意的是，我们在使用自定义模块的量化版替换的时候需要使用 **namedtuple** 这种形式。
*   **quant\_map**：存储最终的量化模块，着包括 pytorch\_quantization 默认的量化模块和用户提供的自定义量化模块

**prepare\_state 方法**

*   这个方法是用于准备 **monkey patching** 机制中使用的量化模块列表
*   它接受两个参数：**float\_module\_list** 和 **custom\_map**
*   **float\_module\_list** 是用户指定的不应该被替换的模块列表
*   **custom\_map** 是用户提供的除了 **torch.nn** 之外的自定义模块量化版本的映射
*   该方法首先基于 **default\_quant\_map** 生成 **quant\_map**，但会跳过 **float\_module\_list** 中指定的模块
*   然后，它会将 **custom\_map** 中的自定义模块添加到 **quant\_map** 中
*   同时，它也会在 **orginal\_func\_map** 中存储原始模块，以便以后可以恢复。

**apply\_quant\_modules 方法**

*   这个用于实际应用 **monkey patching**
*   它会遍历 **quant\_map** 中注册的模块，将它们替换为量化版本，并在 **orginal\_func\_map** 中存储原始模块，以便以后恢复
*   我们可以在运行时动态地替换 torch.nn 中的模块，将其变为量化版本，从而实现模型的量化。

**restore\_float\_modules 方法**

*   这个方法用于恢复原始模块，即撤销之前应用的 **monkey patching**
*   它会遍历 **orginal\_func\_map**，将原始模块替换回去

综上，**QuantModuleReplacementHelper** 类是一个重要的辅助类，用于实现模块的动态替换，以便进行[模型量化](https://so.csdn.net/so/search?q=%E6%A8%A1%E5%9E%8B%E9%87%8F%E5%8C%96&spm=1001.2101.3001.7020)。通过这个类，用户可以灵活地指定哪些模块应该被量化，哪些不应该被量化，甚至可以提供自定义的量化模块，为我们提供了一种高效且灵活的方式来替换模型的量化版本。

那下面我们就来具体看看量化版本的模块到底是如何实现的，我们以 **QuantConv2d** 为例说明

首先 **QuantConv2d** 继承自 **\_QuantConvNd**，而 **\_QuantConvNd** 又继承自 **torch.nn.modules.conv.\_ConvNd** 和 **\_utils.QuantMixin**，那我们重点来关注下 **QuantMixin** 类的工作流程

**QuantMixin** 类的定义如下：

```python
class QuantMixin():
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_weight(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_weight = copy.deepcopy(value)

    def init_quantizer(self, quant_desc_input, quant_desc_weight, num_layers=None):
        """Helper function for __init__ of quantized module

        Create input and weight quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            num_layers: An integer. Default None. If not None, create a list of quantizers.
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        self._fake_quant = True
        if (not quant_desc_input.fake_quant) or (not quant_desc_weight.fake_quant):
            raise ValueError("Only fake quantization is supported!")

        logging.info("Input is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_input.fake_quant else "fake ",
                     quant_desc_input.num_bits, self.__class__.__name__, quant_desc_input.axis)
        logging.info("Weight is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_weight.fake_quant else "fake ",
                     quant_desc_weight.num_bits, self.__class__.__name__, quant_desc_weight.axis)

        if num_layers is None:
            self._input_quantizer = TensorQuantizer(quant_desc_input)
            self._weight_quantizer = TensorQuantizer(quant_desc_weight)
        else:
            self._input_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_input) for _ in range(num_layers)])
            self._weight_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_weight) for _ in range(num_layers)])

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer
    # pylint:enable=missing-docstring
```

它是一个混合类，用于向量化模块和基本的量化逻辑，它的结构和功能如下：

**类属性**

*   **default\_quant\_desc\_input**: 输入张量的默认量化描述符。
*   **default\_quant\_desc\_weight**: 权重张量的默认量化描述符。

**set\_default\_quant\_desc\_input(weight) 类方法**

*   这两个方法用于设置输入和权重张量的自定义描述符
*   它们接受一个 **QuantDescriptor** 实例作为参数，并将其复制为对应的默认描述符

**init\_quantizer 方法**

*   这是一个辅助方法，通常在量化模块的 **\_\_init\_\_** 方法中调用
*   它会根据提供的量化描述符创建输入和权重量化器，通过 **TensorQuantizer** 来创建

值得注意的是描述符是 **ScaleQuantDescriptor** 类的实例，**ScaleQuantDescriptor** 类的描述如下：

```python
class ScaledQuantDescriptor():
    """Supportive descriptor of quantization

    Describe how a tensor should be quantized. A QuantDescriptor and a tensor defines a quantized tensor.

    Args:
        num_bits: An integer. Number of bits of quantization. It is used to calculate scaling factor. Default 8.
        name: Seems a nice thing to have

    Keyword Arguments:
        fake_quant: A boolean. If True, use fake quantization mode. Default True.
        axis: None, int or tuple of int. axes which will have its own max for computing scaling factor.
            If None (the default), use per tensor scale.
            Must be in the range [-rank(input_tensor), rank(input_tensor)).
            e.g. For a KCRS weight tensor, quant_axis=(0) will yield per channel scaling.
            Default None.
        amax: A float or list/ndarray of floats of user specified absolute max range. If supplied,
            ignore quant_axis and use this to quantize. If learn_amax is True, will be used to initialize
            learnable amax. Default None.
        learn_amax: A boolean. If True, learn amax. Default False.
        scale_amax: A float. If supplied, multiply amax by scale_amax. Default None. It is useful for some
            quick experiment.
        calib_method: A string. One of ["max", "histogram"] indicates which calibration to use. Except the simple
            max calibration, other methods are all hisogram based. Default "max".
        unsigned: A Boolean. If True, use unsigned. Default False.

    Raises:
        TypeError: If unsupported type is passed in.

    Read-only properties:
        - fake_quant:
        - name:
        - learn_amax:
        - scale_amax:
        - axis:
        - calib_method:
        - num_bits:
        - amax:
        - unsigned:
    """

    def __init__(self, num_bits=8, name=None, **kwargs):
        ...
```

它描述了张量应该如何进行量化，这个类定义了量化所需的参数和属性，提供了一种灵活的方式来配置量化过程，它的结构和功能如下：

**类属性**

*   **num\_bits**：量化位数，用于计算缩放因子
*   **fake\_quant**：伪量化模式，如果设置为 True，则使用伪量化，默认为 True
*   **axis**：用于计算缩放因子 scale 的轴，如果为 None，则使用每个张量计算 scale，例如 input\_quant；如果等于 0 将按照每个通道计算 scale，默认为 None
*   **amax**：动态范围的最大值，如果用户提供，则使用该值进行量化
*   **learn\_amax**：布尔值，如果为 True 则将学习 amax，默认 False
*   **scale\_amax**：如果用户提供，则会将 amax 乘以 scale\_amax，默认 None
*   **calib\_method**：校准方法，可以是 Max 最大值校准或者 Histogram 直方图校准，默认直方图校准

而量化器模块 **TensorQuantizer** 类的描述如下：

```python
class TensorQuantizer(nn.Module):
    """Tensor quantizer module

    This module uses tensor_quant or fake_tensor_quant function to quantize a tensor. And wrappers variable, moving
    statistics we'd want when training a quantized network.

    Experimental features:
        ``clip`` stage learns range before enabling quantization.
        ``calib`` stage runs calibration

    Args:
        quant_desc: An instance of :func:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
        disabled: A boolean. If True, by pass the whole module returns input. Default False.
        if_quant: A boolean. If True, run main quantization body. Default True.
        if_clip: A boolean. If True, clip before quantization and learn amax. Default False.
        if_calib: A boolean. If True, run calibration. Not implemented yet. Settings of calibration will probably
            go to :func:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.

    Raises:

    Readonly Properties:
        - axis:
        - fake_quant:
        - scale:
        - step_size:

    Mutable Properties:
        - num_bits:
        - unsigned:
        - amax:
    """

    # An experimental static switch for using pytorch's native fake quantization
    # Primary usage is to export to ONNX
    use_fb_fake_quant = False

    def __init__(self, quant_desc=QuantDescriptor(), disabled=False, if_quant=True, if_clip=False, if_calib=False):
        """Initialize quantizer and set up required variables"""
        ...
```

该类是我们实际的张量量化模块，即量化器模块。它使用 tensor\_quant 或者 fake\_tensor\_quant 函数对张量进行量化，特点是在启动量化之前它会计算量化的一个动态范围，之后根据我们选用的校准方法来进行校准。它的结构和功能如下：

**\_\_init\_\_ 方法**

*   接收一个 **QuantDescriptor** 量化描述符实例作为参数，用于设置量化的各种属性和参数
*   **disabled** 参数用于控制是否禁用该层进行量化，默认 False
*   **if\_quant** 参数用于控制是否运行主体量化逻辑，默认 True
*   **if\_clip** 参数用于控制是否在量化前裁剪并学习 amax，默认 False
*   **if\_calib** 参数控制是否运行校准，默认 False

OK！以上就是 QDQ 节点的自动插入和 initializer 函数的简单分析，下面我们来介绍下手动插入 QDQ 量化节点。

## 3.2 手动插入QDQ节点

上面我们对 initializer 函数进行了简单的分析，其中涉及了多个类之间的继承关系，下面是 initializer 函数中涉及到的类的继承关系图：

![d266ac5e9fe31c1b7912b94edc202afb](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/d266ac5e9fe31c1b7912b94edc202afb.png)

大家可以根据上面的继承关系图对应到实际的代码中去看看相应的继承关系，比如在量化模块转换中 **QuantConv2d** 继承自 **\_QuantConvNd**，而 **QuantConv2d** 与 **Conv2d** 又可以相互转换。

下面我们就根据上面的继承关系流程图来手动实现 QDQ 节点的插入，我们主要是实现三个函数：

**1.** **replace\_to\_quantization\_model**

```python
def replace_to_quantization_model(model):
    
    module_list = {}
    
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)  # module -> torch.nn.modules.conv.Conv1d
        module_list[id(module)] = entry.replace_mod
    
    torch_module_find_quant_module(model, module_list)
```

该函数是手动插入量化节点的起始函数，目的是为模型生成一个替换映射，并调用[递归](https://edu.csdn.net/course/detail/40020?utm_source=glcblog&spm=1001.2101.3001.7020)函数来查找和替换模型中的模块。

*   **生成替换映射**：module\_list 字典基于 quant\_modules.\_DEFAULT\_QUANT\_MAP 构建，它包含了原始 pytorch 模块类（如 nn.Conv2d）到它们对应的量化模块类（如 quant\_modules.QuantConv2d）的映射。
*   **调用递归函数**：调用 torch\_module\_find\_quant\_module 函数开始递归过程，传入模型和 module\_list 作为参数

**2.** **torch\_module\_find\_quant\_module**

```python
def torch_module_find_quant_module(model, module_list, prefix=''):
    for name in model._modules:
        submodule = model._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_list, prefix=path) # 递归

        submodule_id = id(type(submodule))
        if submodule_id in module_list:
            # 转换
            model._modules[name] = transfer_torch_to_quantization(submodule, module_list[submodule_id])
```

该函数递归的遍历模型中的所有子模块，寻找可以被量化版本替换的模块。

*   **递归遍历**：对于模型中的每一个子模块，函数递归调用自身以遍历更深层次的子模块
*   **检查并替换模块**：如果子模块的类型存在于 module\_list 映射中，则使用 transfer\_torch\_to\_quantization 函数将子模块替换为其量化版本
*   **路径记录**：prefix 参数用于跟踪当前子模块在模型中的路径

**3.** **transfer\_torch\_to\_quantization**

```python
def transfer_torch_to_quantization(nn_instance, quant_module):
    
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
```

核心函数，用于将一个标准的 pytorch 模块实例转换为其量化版本

*   **实例量化模块**：使用 quant\_module 参数（一个量化模块类）创建一个新的量化模块实例
*   **属性复制**：将原始模块 nn\_instance 的所有属性复制到新的量化模块实例中
*   **初始化量化器**：通过 通过 \_\_init\_\_ 内部函数调用，使用 init\_quantizer 方法初始化量化器。这个步骤确保量化模块具有适当的量化描述符，并设置了任何必要的量化参数。
*   **启用直方图校准优化**: 如果量化器使用 HistogramCalibrator，将设置 \_torch\_hist 标志以加速直方图计算过程。

完整的示例代码如下：

```python

import torch
from models.yolo import Model
from pytorch_quantization import calib
from pytorch_quantization import quant_modules
from pytorch_quantization.nn.modules import _utils as quant_nn_utils

def load_yolov7_model(weight, device='cpu'):
    ckpt  = torch.load(weight, map_location=device)
    model = Model("cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model

def prepare_model(weight, device):
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()    # conv bn 进行层的合并, 加速
    return model

def transfer_torch_to_quantization(nn_instance, quant_module):
    
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
            model._modules[name] = transfer_torch_to_quantization(submodule, module_list[submodule_id])
        
def replace_to_quantization_model(model):
    
    module_list = {}
    
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)  # module -> torch.nn.modules.conv.Conv1d
        module_list[id(module)] = entry.replace_mod
    
    torch_module_find_quant_module(model, module_list)


if __name__ == "__main__":

    weight = "yolov7.pt"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pth_model = load_yolov7_model(weight, device)

    model = prepare_model(weight, device)
    replace_to_quantization_model(model)
    print(model)
```

运行效果如下：

![a2e73bee11aec11e9558c33f98da3d88](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/a2e73bee11aec11e9558c33f98da3d88.png)

**图3-3 手动插入QDQ节点模型**

可以看到对应的 pytorch 算子都变成了其对应的量化版本，同时还拥有了 \_input\_quantizer 和 \_weight\_quantizer。

下面我们简单总结下手动插入量化节点的流程：

*   **1\. 准备模型**: 在模型准备阶段，执行必要的优化（如层融合）。
    
*   **2\. 构建替换映射**: 基于预定义的量化映射（quant\_modules.\_DEFAULT\_QUANT\_MAP），创建一个映射字典，用于将标准模块替换为相应的量化模块。
    
*   **3\. 递归遍历模型**: 使用 torch\_module\_find\_quant\_module 函数递归地遍历整个模型的子模块。
    
*   **4\. 检查并替换模块**: 对于每个子模块，如果其类型在替换映射中，则使用 transfer\_torch\_to\_quantization 函数将其替换为量化版本。
    
*   **5\. 初始化量化器**: 在量化模块实例中初始化量化器，确保量化描述符和校准器设置正确。
    
*   **6\. 优化量化过程**: 如果使用直方图校准器，开启 \_torch\_hist 以加快直方图校准过程。
    
*   **7\. 插入量化节点**: 将量化模块实例插入到模型中，替换原始的非量化模块。
    
*   **8\. 完成量化准备**: 完成这些步骤后，模型就被转换为一个量化模型，其中包含了为推断和/或训练过程量化准备好的节点。
    

通过这个过程，模型中的每个符合条件的模块都会被其量化版本所替换。这种手动插入量化节点的方法可以让你有更细粒度的控制，例如在模型的特定部分使用不同的量化策略或描述符。

OK！以上就是 QDQ 节点的手动插入，下面我们来介绍下手动 initialize。

## 3.3 手动initialize

我们先来看下手动插入量化节点后模型 mAP 测试的结果，运行如下图所示：

![835dc14ca12e4382ef1d6173038e9d7a](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/835dc14ca12e4382ef1d6173038e9d7a.png)

可以看到测试结果和 torch 差不多，此外量化器使用的是默认 MaxCalibrator 校准器，其中的 initialize 是按照**默认的方法**去进行初始化的。

下面我们手动来实现 initialize，不使用默认的 Max 量化器而是去使用直方图，我们来看下应该怎么去操作。

**initialize** 函数的实现代码如下：

```python
from absl import logging as quant_logging
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor

# input: Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)
```

在代码中我们将 QuantConv2d、QuantMaxPool2d 以及 QuantLinear 的 input 量化器修改为了直方图，我们可以自己手动编写 initialize 从而完成一些自定义的操作。

接下来我们来进行验证，看看量化方式会不会有所变化，运行效果如下：

![d38812303c139eff477729e0a7c2c95f](TensorRT量化实战课YOLOv7量化：YOLOv7-PTQ量化(一)/d38812303c139eff477729e0a7c2c95f.png)

我们可以看到 input 的量化器使用的是直方图，权重依旧使用的是默认的 Max 量化器，这说明我们手动实现的 initialize 起作用了。另外可以看到 mAP 和之前的没有什么变化，因为我们并没有真正的去执行相关量化操作，而仅仅插入了量化节点。

下节课我们将会讲解模型标定的相关内容。

# 总结

> 本次课程介绍了 YOLOv7-PTQ 量化流程中的准备工作和 QDQ 节点的插入，其中准备工作包括模型和数据集的准备，而量化节点的插入我们介绍了自动和手动插入两种方式，其中自动插入是调用 initialize 函数来完成的，而手动插入是根据 initialize 函数中类的继承关系图来一步步实现的。QDQ 节点插入后从模型结构可以看到每个节点多了 \_input\_quantizer 和 \_weight\_quantizer 两个输入，同时 torch 模块也变成了对应的量化版本。最后我们手动实现了 initialize 函数将 input 的校准器从默认的 Max 替换成了直方图。
> 
> 下节课程我们将会真正的去编写代码对插入量化节点的模型进行标定计算以完成 PTQ 模型的量化和导出工作。

本文转自 <https://blog.csdn.net/qq_40672115/article/details/134108526>，如有侵权，请联系删除。