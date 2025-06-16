# 简介

关于orin的算力细节没查到，这里总结一下

关于orin算力的一些描述

* drive orin 官网标称算力为TOPS(INT8)|5.2TOPS(FP32)  其中是包含了cuda core 和tensor core 的计算能力。但是tensor core只能做卷积和矩阵乘。
* 因为nv的GPU是一个通用型的GPU，支持的网络多，但是一般情况下一个网络模型不能发挥orin的全部算力。发挥算力最好的是ResNeT34，也只能发挥出芯片全部算力的60% ~67%的。优化得好的网络模型可以发挥30%~40%的算力。一般情况下一个网络模型可以发挥25%~30%的算力。
* 如何充分利用芯片的算力：
  * 模型选型方面：
    * 每个模型都使用INT8 pref一下。关注一下几个方面
    * 看耗时大的模型
    * 看单个模型耗时最多的Layer，进行优化
    * 换BackBone
    * 主要压榨模型的算力
  * 模型部署量化方面：
    * 查看网络是否是都INT8（包含是IN OUT都是INT8）
    * 看qdq节点是否合适
    * 具体看耗时多的Layer
    * 没有什么好的手段可以分析并规划模型的算力，只能看模型运行当前的帧率是否可以满足需求来进行优化。
    * tegrastats获取的GPU利用只是 每秒抓取6次GPU的使用情况做平均，只要GPU在使用就会被抓取，并不能真是反应GPU的具体使用率。
    * 在nsigth system中，获取SM Activate也不是非常准确的反应当前的GPU利用率。因为只要是一个SM中有活跃的warp就被统计为使用了，因此也不是一个精准的指标，NV并没有一个可以使用的工具来真是反应当前的GPU占用情况的工具。但是SM Activate参考了。
* orin 的算力描述是167TOPS（INT8），包含了cuda core 和 tensor core 的算力。但是没有说明分别是多少，大概cuda core只有十几TOPS。剩下的大部分算力都是tensor core的算力。
  orinx：

| One Orin SoC | Deep Learning Accelerators (DLA)                | 87 TOPS (INT8)  |
| ------------ | ----------------------------------------------- | --------------- |
|              | NVIDIA Ampere architecture-class integrated GPU | 167 TOPS (INT8) |

●如何充分利用tensor core 算力：tensor core主要是进行矩阵和卷积运算。网络结构多是矩阵和卷积可以充分利用tensor core