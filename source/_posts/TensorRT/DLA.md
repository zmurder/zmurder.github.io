# 简介

介绍一些DLA的相关知识点，持续完善，想到哪里写到哪里。

# I/O Formats on DLA

官方的描述如下：

DLA supports formats that are unique to the device and have constraints on their layout due to vector width byte requirements.

For DLA input tensors, `kDLA_LINEAR(FP16, INT8)`, `kDLA_HWC4(FP16, INT8)`, `kCHW16(FP16)`, and `kCHW32(INT8)` are supported.

For DLA output tensors, only `kDLA_LINEAR(FP16, INT8)`, `kCHW16(FP16)`, and `kCHW32(INT8)` are supported.

For `kCHW16` and `kCHW32` formats, if `C` is not an integer multiple, it must be padded to the next 32-byte boundary.

For `kDLA_LINEAR` format, the stride along the `W` dimension must be padded up to 64 bytes. The memory format is equivalent to a `C` array with dimensions `[N][C][H][roundUp(W, 64/elementSize)]` where `elementSize` is `2` for FP16 and `1` for Int8, with the tensor coordinates `(n, c, h, w)` mapping to array subscript `[n][c][h][w]`.

For `kDLA_HWC4` format, the stride along the `W` dimension must be a multiple of 32 bytes on Xavier and 64 bytes on NVIDIA Orin.

- When `C == 1`, TensorRT maps the format to the native grayscale image format.
- When `C == 3` or `C == 4`, it maps to the native color image format. If `C == 3`, the stride for stepping along the W axis must be padded to `4` in elements.
  - In this case, the padded channel is located at the 4th index.  Ideally, the padding value does not matter because the DLA compiler  paddings the 4th channel in the weights to zero; however, it is safe for the application to allocate a zero-filled buffer of four channels and  populate three valid channels.
- When `C` is `{1, 3, 4}`, then padded `C'` is `{1, 4, 4}` respectively, the memory layout is equivalent to a `C` array with dimensions `[N][H][roundUp(W, 32/C'/elementSize)][C']` where `elementSize` is `2` for FP16 and `1` for Int8. The tensor coordinates `(n, c, h, w)` mapping to array subscript `[n][h][w][c]`, `roundUp` calculates the smallest multiple of `64/elementSize` greater than or equal to `W`.

When using `kDLA_HWC4` as the DLA input format, it has the following requirements:

- `C` must be `1`, `3`, or `4`
- The first layer must be convolution.
- The convolution parameters must meet DLA requirements. For more information, refer to the [DLA Supported Layers and Restrictions](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html#dla-layers-restrictions-overview) section.

When GPU fallback is enabled, TensorRT may insert reformatting layers to meet the DLA requirements. Otherwise, the input and output formats  must be compatible with DLA. In all cases, the strides that TensorRT  expects data to be formatted with can be obtained by querying `IExecutionContext::getStrides`.



输入和输出有一定的格式要求，但是什么样的格式是最有的呢？

 参考 https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-825/pdf/TensorRT-Developer-Guide.pdf 里面有这么一段话

```txt
Q: What is the best practice to use reformat-free network I/O tensors for DLA? A: First,
you have to check if your network can run entirely on DLA, then try to build the network
by specifying kDLA_LINEAR, kDLA_HWC4 or kCHW16/32 format as allowed I/O formats.
If multiple formats can work, you can profile them and choose the fastest I/O format
combination. If your network indeed performs better with kDLA_HWC4, but it doesn't
work, you have to check which requirement listed in the previous section is unsatisfied
```

意思是自己尝试一下什么格式速度最快。

但是根据官网的描述有一些经验吧，也就是什么使用用kDLA_LINEAR 什么时候用kDLA_HWC4之类的

* kDLA_HWC 专为图像处理优化，**维度顺序**：`NHWC`（Batch×Height×Width×Channels），适合相机输入（上面提到了The first layer must be convolution.）
* kDLA_LINEAR是DLA支持的行主序（Row-Major）内存布局，**维度顺序**：`NCHW`（Batch×Channels×Height×Width），适合非结构化数据，例如激光雷达点云数据



# 附录：

* 官方文档 [Working with DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html)