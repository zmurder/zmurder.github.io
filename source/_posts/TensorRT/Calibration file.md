# 1 简介 

我们在使用Tensorrt的隐形量化时，需要生成一个cache文件，用于onnx生成engine文件使用。如果我们使用`trtexec`来将onnx文件生成为和我们的GPU相关的隐形量化后的engine文件时需要参数 参考[A.2.1.2. Serialized Engine Generation](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#trtexec-serialized-engine)   [A.2.1.4. Commonly Used Command-line Flags](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#trtexec-flags)  

例如

```shell
trtexec --onnx=test.onnx --calib=test.cache --int8 --saveEngine=test.engine
```

上面的命令就是使用到了隐形量化的cahe文件。这个文件如何生成的参考我之前的文章     [TensorRT INT8量化代码](https://zmurder.github.io/TensorRT/TensorRT%20INT8%E9%87%8F%E5%8C%96%E4%BB%A3%E7%A0%81/)        

这里只是补充说明一下生成的cache文件内容

参考https://www.ccoderun.ca/programming/doxygen/tensorrt/md_TensorRT_samples_opensource_sampleINT8_README.html#calibration-file

一个校准文件（就是我们上面的test.cache文件）存储每个网络张量的激活量表。激活量表是使用从校准算法生成的动态范围来计算的，换句话说，`ABS（max_dynamic_range） / 127.0f`。

校准文件名为`CalibrationTable<NetworkName>`，其中`<NetworkName]`是您的网络名称，例如`mnist`。该文件位于`TensorRT-x.x.x.x/data/mnist`目录中，其中`x.x.x.x`是您安装的`TensorRT`版本。

`CalibrationTable`内容包括：

```shell
TRT-7000-EntropyCalibration2
data: 3c008912
conv1: 3c88edfc
pool1: 3c88edfc
conv2: 3ddc858b
pool2: 3ddc858b
ip1: 3db6bd6e
ip2: 3e691968
prob: 3c010a14
```

这里：

`<TRT xxxx>-<xxxxxxx>`TensorRT版本后面是校准算法，例如`EntropyCalibration2`。

`<layer name>`：值对应于网络中每个张量校准期间确定的浮点激活标度。

`CalibrationTable`文件是在运行校准算法的构建阶段生成的。创建校准文件后，可以在后续运行时读取该文件，而无需再次运行校准。您可以为`readCalibrationCache()`提供实现，以便从所需位置加载校准文件。如果读取的校准文件与校准器类型(用于生成文件)和TensorRT版本兼容，则构建器将跳过校准步骤并使用校准文件中的每个张量尺度值。

## 附录：

* ### trtexec command-line flags [A.2.1.4. Commonly Used Command-line Flags](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#trtexec-flags)

* https://www.ccoderun.ca/programming/doxygen/tensorrt/md_TensorRT_samples_opensource_sampleINT8_README.html#calibration-file