# 前言

这里总结一些经常使用到的HPC的经验。便于后期对照查询使用

# TensorRT辅助流

参考 https://developer.nvidia.com/docs/drive/drive-os/6.0.8/public/drive-os-tensorrt/pdf/NVIDIA-TensorRT-8.6.11-API-Reference-for-DRIVE-OS.pdf

如果希望stream是可控的，最好将辅助流设置为0.

如是是使用代码构建engine：

* 函数为 `setMaxAuxStreams()`。如果不设置，那么在分析nsys时会发现TensorRT构建的engine有可能会有多个stream。

如果是使用trtexec构建engine

* **在构建engine的时候使用trtexec的选项--maxAuxStreams=N 设置为--maxAuxStreams=0 即可。**

在我们多线程，多stream的程序中，最好是可以手动控制stream。因此这里设置为0最好。



# trtexec的IO format

在使用trtexec构建engine是可以指定engine的输出输出格式，这在一个onnx拆分为两个联接在一起的engine时非常有用，因为在拆分的部分可能存在reformat节点，在两个engine都存在比必要的reformat节点会增加延迟，这时只需要指定第一个engine的输出的format和第二个engine的输入formart就可以了

trtexec对应的选项是--inputIOFormats  和 --outputIOFormats

这里根据官方的文档，如果engine的第一层是卷积层，那么最合适的format是**Tensor Core 实现的卷积需要 NHWC 布局，并且在 NHWC 中布局输入张量时速度最快**。参考参考[Tensor Layouts In Memory: NCHW vs NHWC](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout)

但是如果engine是在DLA上运行，那么参考https://zmurder.github.io/TensorRT/DLA/?highlight=hwc

- kDLA_HWC 专为图像处理优化，**维度顺序**：`NHWC`（Batch×Height×Width×Channels），适合相机输入（上面提到了The first layer must be convolution.）
- kDLA_LINEAR是DLA支持的行主序（Row-Major）内存布局，**维度顺序**：`NCHW`（Batch×Channels×Height×Width），适合非结构化数据，例如激光雷达点云数据

参考https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/advanced.html

| Format        | `kINT32`     | `kFLOAT`     | `kHALF`                               | `kINT8`      | `kBOOL` | `kUINT8` | `kINT64` | `BF16`       | `FP8`        | `FP4/INT4` |
| ------------- | ------------ | ------------ | ------------------------------------- | ------------ | ------- | -------- | -------- | ------------ | ------------ | ---------- |
| `kLINEAR`     | Only for GPU | Yes          | Yes                                   | Yes          | Yes     | Yes      | Yes      | Yes          | Yes          | Yes        |
| `kCHW2`       | No           | No           | Only for GPU                          | No           | No      | No       | No       | Yes          | No           | No         |
| `kCHW4`       | No           | No           | No                                    | Yes          | No      | No       | No       | No           | No           | No         |
| `kHWC8`       | No           | No           | Only for GPU                          | No           | No      | No       | No       | Only for GPU | No           | No         |
| `kCHW16`      | No           | No           | Only for DLA                          | No           | No      | No       | No       | No           | No           | No         |
| `kCHW32`      | No           | Only for GPU | Only for GPU                          | Yes          | No      | No       | No       | No           | No           | No         |
| `kDHWC8`      | No           | No           | Only for GPU                          | No           | No      | No       | No       | Only for GPU | No           | No         |
| `kCDHW32`     | No           | No           | Only for GPU                          | Only for GPU | No      | No       | No       | No           | No           | No         |
| `kHWC`        | No           | Only for GPU | No                                    | No           | No      | Yes      | No       | No           | No           | No         |
| `kDLA_LINEAR` | No           | No           | Only for DLA                          | Only for DLA | No      | No       | No       | No           | No           | No         |
| `kDLA_HWC4`   | No           | No           | Only for DLA                          | Only for DLA | No      | No       | No       | No           | No           | No         |
| `kHWC16`      | No           | No           | Only for NVIDIA Ampere GPUs and later | Only for GPU | No      | No       | No       | No           | Only for GPU | No         |
| `kDHWC`       | No           | Only for GPU | No                                    | No           | No      | No       | No       | No           | No           | No         |



# CPU内存类型

在CPU生申请的的内存主要和GPU交互，例如H2D和D2H的拷贝，这里只说明一种情况，就是当我们的进程收到另一个进程发来的一个内存指针，需要将这段内存放在GPU中运算。

我们都知道CPU与GPU交互的内存最好都是pinne mem，这样拷贝速度最快。但是上面的情况是CPU的内存空间是我们的进程收到的，不是我们自己申请的，如何保证这段内存是pinned mem呢？我们可以使用`cudaHostRegister`函数来处理这段内存，从pageable mem变为pinned mem。（只处理一次就可以了）。

内存相关的细节参考

# cuda context

一般情况下我们的一个进程就是共用一个cuda context，在多个进程使用到GPU时会涉及到cuda context的切换，切换时间大约也就是1ms以内。但是有时候会因为切换context造成一些GPU延迟的不稳定。这里举一个例子说明：

假如我们有两个进程使用GPU，A和B进程。

context A->context B 切换，那么A的kernel这时候可能还没有运算完成，但是context切换到B了，GPU同一时刻只有一个cuda context可以运行，因此A的这个kernel就会暂停运行，直到重新切换回context A才能继续运行。

这时我们分析nsys时会发现这个contextA的运行时间是不稳定的（或者kernel的时间是不稳定的）。可能就是这个原因。

# kernel或者enqueue时间不稳定

除了上面说到的cuda context的切换会导致kernel或者模型的推理enqueue时间不稳定，stream的并行数量设置也是导致时间不稳定的原因。这个有一个突出的特点就是GPU空闲，但是kernel偶发时间变长。

使用环境变量设置stream的并发数 export CUDA_DEVICE_MAX_CONNECTIONS=32等 默认应该是8.

如图cudagraph或者一个kernel都运行完成了，但是从nsys上分析sync等了一会儿才返回，这时就要考虑是不是stream不够了。可以根据自己的线程和需要的stream设置需要的并发stream数量。