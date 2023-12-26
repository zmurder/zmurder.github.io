# 简介

这里列出自己遇见的一些cuda报错，方便后期再次遇见解决



* no kernel image is available for execution on device
  * 我这里是设置的`-gencode arch=compute_61,code=sm_61`与自己的显卡不匹配导致的，需要设定为自己显卡的算里数字，参考[GPU架构与算力](https://zmurder.github.io/GPU/GPU%E6%9E%B6%E6%9E%84%E4%B8%8E%E7%AE%97%E5%8A%9B/) 和[官网](https://developer.nvidia.com/cuda-gpus)