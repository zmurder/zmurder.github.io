Nvidia diver是最基础的跟硬件直接交互的底层软件，cuda依赖于driver，cuDNN依赖于cuda，tensorRT最终模型的推理加速依赖于前面这些基础的加速环境。



搜索显卡

```shell
 lspci | grep VGA     # 查看集成显卡
 lspci | grep NVIDIA  # 查看NVIDIA显卡
```

# 1 GPU驱动安装

ubuntu显卡驱动安装有四中方法：

- 通过ubuntu仓库安装
- 在英伟达官网选择相应版本的驱动安装
- 在ubuntu软件和更新界面的附加驱动中安装
- -添加ppa源安装

## 方法一：ubuntu仓库安装

只需要一条指令即可安装成功，成功安装后需要重启

```shell
sudo ubuntu-drivers autoinstall
```

```bash
nvidia-smi  #若出现电脑GPU信息则成功
```

## 方法二：英伟达官网安装

首先进入[英伟达官网](https://www.nvidia.com/Download/index.aspx?lang=en-us)

## 方法三：附加驱动安装方法

更新软件源

![image-20220905145910202](ubuntu下的GPU环境配置.assets/image-20220905145910202.png)

 在其他站点中选择清华源，更新后终端执行

```bash
sudo apt-get update
sudo apt-get upgrade
```

安装驱动：菜单栏选择附加驱动，会进行自动搜索，选择一个版本的专有驱动，点击应用更改，更改后重启即可，

## 方法四：ppa仓库安装

（1）首先禁用nouveau

```bash
sudo gedit /etc/modprobe.d/blacklist.conf
```

在最后一行添加

```bash
blacklist nouveau
options nouveau modeset=0 #禁用nouveau第三方驱动 本质就是禁用集显
```

（2）执行以下指令

```shell
sudo apt-get remove --purge nvidia*
sudo update -initramfs -u # 更新内核 这个最好不要随便执行，先跳过
sudo add-apt-repository ppa:graphics-drivers/ppa # 添加ppa源
sudo apt-get update
sudo apt-get install nvidia-driver-450 # 这里版本可以根据自己需求来
sudo apt-get install mesa-common-dev
sudo apt-get update
sudo apt-get upgrade
nvidia-smi # 用于确认是否安装成功
sudo sed -i "s/ppa\.launchpad\.net/lanuchpad.moruy.cn/g" /etc/apt/sources.list.d/*.list #ppa加速
```

# 2 深度学习环境

## 2.1 cuda安装

### 2.1.1 下载cuda

[下载链接](https://developer.nvidia.com/cuda-toolkit-archive)

这里下载的是11.4.0

![image-20220905150820507](ubuntu下的GPU环境配置.assets/image-20220905150820507.png)

按照官方的提示下载并运行run文件

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.runsudo sh cuda_11.4.0_470.42.01_linux.run
```

按装出现下面提示，输入accept

![image-20220905150902890](ubuntu下的GPU环境配置.assets/image-20220905150902890.png)

由于上面已经安装过显卡的驱动了。这里把驱动安装取消（空格键）其他都选择上

![image-20220905150946443](ubuntu下的GPU环境配置.assets/image-20220905150946443.png)

![image-20220905151246979](ubuntu下的GPU环境配置.assets/image-20220905151246979.png)

### 2.1.2 设置权限

```bash
sudo chmod -R 755 /usr/local/cuda-11.4
```

安装完成后会有一个连接文件`/usr/local/cuda` 指向真正的`/usr/local/cuda-11.4`

![image-20220905152752889](ubuntu下的GPU环境配置.assets/image-20220905152752889.png)

### 2.1.3 配置环境变量

可以在~/.bashrc中加入环境变量

```shell
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

2.1.4 验证安装

```shell
nvcc -V
```

![image-20220905151422900](ubuntu下的GPU环境配置.assets/image-20220905151422900.png)

版本信息

```shell
cat /usr/local/cuda/version.json
```

![image-20220905154420841](ubuntu下的GPU环境配置.assets/image-20220905154420841.png)

## 2.2 cuDNN安装

### 2.2.1 cuDNN下载

[下载地址](https://developer.nvidia.com/rdp/cudnn-archive)

这里选择的是cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz

![image-20220905153547803](ubuntu下的GPU环境配置.assets/image-20220905153547803.png)

### 2.2.2 cudNN安装

```shell
tar -xvf cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz 
cd cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive
```

```shell
sudo cp include/cudnn*.h /usr/local/cuda-11.4/include 
sudo cp -P lib/libcudnn* /usr/local/cuda-11.4/lib64 
sudo chmod a+r /usr/local/cuda-11.4/include/cudnn*.h /usr/local/cuda-11.4/lib64/libcudnn*
sudo chmod -R 755 /usr/local/cuda-11.4
```

2.2.3 cuDNN版本查看

```shell
cat /usr/local/cuda/include/cudnn_version.h 
```

![image-20220905154205792](ubuntu下的GPU环境配置.assets/image-20220905154205792.png)

# TensorRT安装

EA 版本代表抢先体验（在正式发布之前）。
GA 代表通用性。 表示稳定版，经过全面测试。

# 附录：

[显卡驱动安装](https://blog.csdn.net/lixushi/article/details/118575942)

[cuda安装官方](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)

[cudnn安装官方](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#prerequisites)

[博客](https://zhuanlan.zhihu.com/p/540588163#:~:text=Nvidia%20diver%E6%98%AF%E6%9C%80%E5%9F%BA%E7%A1%80%E7%9A%84%E8%B7%9F%E7%A1%AC%E4%BB%B6%E7%9B%B4%E6%8E%A5%E4%BA%A4%E4%BA%92%E7%9A%84%E5%BA%95%E5%B1%82%E8%BD%AF%E4%BB%B6%EF%BC%8Ccuda%E4%BE%9D%E8%B5%96%E4%BA%8Edriver%EF%BC%8CcuDNN%E4%BE%9D%E8%B5%96%E4%BA%8Ecuda%EF%BC%8CtensorRT%E6%9C%80%E7%BB%88%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%BE%9D%E8%B5%96%E4%BA%8E%E5%89%8D%E9%9D%A2%E8%BF%99%E4%BA%9B%E5%9F%BA%E7%A1%80%E7%9A%84%E5%8A%A0%E9%80%9F%E7%8E%AF%E5%A2%83%E3%80%82%20%E6%88%91%E4%BB%AC%E8%BF%99%E9%87%8C%E5%AE%89%E8%A3%85driver,%28460.106.00%29%2Bcuda%20%2811.2.0%29%2BcuDNN%20%288.4.0%29%2BTensorRT%20%288.2.1.8%29%E3%80%82)

[算力查询](https://developer.nvidia.com/cuda-gpus)

