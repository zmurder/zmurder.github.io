#  简介

看了官方的说明文档还是有一点云里雾里，这里说明一些使用nsisght systems过程中的经验之谈。

# 下载与安装

如果是单独安装windows或者是linux下的，可以在官网https://developer.nvidia.com/nsight-systems/get-started下载

可以从https://developer.nvidia.com/tools-downloads 下载历史版本

如果使用的是DRIVE设备，例如orin，那么会在DRIVE OS 中包含（应该是安装cuda时安装的）

需要注意的是，如果你在一个设备上抓取的.nsys-repo文件希望在另一个设备上打开，那么抓取和加载回放的nsys版本需要一致。

## 安装目录

默认的安装目录在`/opt/nvidia/nsight-systems`

如果你安装了多个版本的nsys。那么目录会如下

```bash
(base) zyd@P7479785A244:/opt/nvidia/nsight-systems$ ls
2022.4.2  2023.4.4  2024.5.1  2024.7.2
```



例如我的一个目录如下

```bash
(base) zyd@P7479785A244:/opt/nvidia/nsight-systems/2024.5.1/host-linux-x64$ ls
CrashReporter                    libHostCommon.so               libQt6OpenGL.so.6                             libSshClient.so                     libicui18n.so.71
DumpTimeline                     libInjectionCommunicator.so    libQt6OpenGLWidgets.so.6                      libStreamSections.so                libicuuc.so.71
ImportNvtxt                      libInterfaceData.so            libQt6Positioning.so.6                        libSymbolAnalyzerLight.so           libjpeg.so.8
Mesa                             libInterfaceShared.so          libQt6PrintSupport.so.6                       libSymbolDemangler.so               libnvlog.so
NVIDIA_SLA.pdf                   libInterfaceSharedBase.so      libQt6Qml.so.6                                libTelemetryQuadDClient.so          libpapi.so.5
Plugins                          libInterfaceSharedCore.so      libQt6QmlModels.so.6                          libTimelineAssert.so                libparquet.so
PythonFunctionsTrace             libInterfaceSharedLoggers.so   libQt6Quick.so.6                              libTimelineCommon.so                libparquet.so.500
QdstrmImporter                   libLinuxPerf.so                libQt6QuickParticles.so.6                     libTimelineUIUtils.so               libparquet.so.500.0.0
ResolveSymbols                   libNetworkInfo.so              libQt6QuickTest.so.6                          libTimelineWidget.so                libpfm.so.4
Scripts                          libNsysServerProto.so          libQt6QuickWidgets.so.6                       libarrow.so                         libprotobuf-shared.so
libAgentAPI.so                   libNvQtGui.so                  libQt6Sensors.so.6                            libarrow.so.500                     libsqlite3.so.0
libAnalysis.so                   libNvmlWrapper.so              libQt6Sql.so.6                                libarrow.so.500.0.0                 libssh.so
libAnalysisContainersData.so     libNvtxExtData.so              libQt6StateMachine.so.6                       libboost_atomic.so.1.78.0           libssl.so
libAnalysisData.so               libNvtxwBackend.so             libQt6Svg.so.6                                libboost_chrono.so.1.78.0           libssl.so.1.1
libAnalysisProto.so              libProcessLauncher.so          libQt6SvgWidgets.so.6                         libboost_container.so.1.78.0        libstdc++.so.6
libAppLib.so                     libProtobufComm.so             libQt6Test.so.6                               libboost_date_time.so.1.78.0        nsys-ui
libAppLibInterfaces.so           libProtobufCommClient.so       libQt6UiTools.so.6                            libboost_filesystem.so.1.78.0       nsys-ui.bin
libAssert.so                     libProtobufCommProto.so        libQt6WaylandClient.so.6                      libboost_iostreams.so.1.78.0        nsys-ui.desktop.template
libCommonNsysServer.so           libQt6Charts.so.6              libQt6WaylandCompositor.so.6                  libboost_program_options.so.1.78.0  nsys-ui.png
libCommonProtoServices.so        libQt6Concurrent.so.6          libQt6WaylandEglClientHwIntegration.so.6      libboost_python310.so.1.78.0        nvlog.config.template
libCommonProtoStreamSections.so  libQt6Core.so.6                libQt6WaylandEglCompositorHwIntegration.so.6  libboost_regex.so.1.78.0            python
libCore.so                       libQt6DBus.so.6                libQt6WebChannel.so.6                         libboost_serialization.so.1.78.0    reports
libCudaDrvApiWrapper.so          libQt6Designer.so.6            libQt6WebEngineCore.so.6                      libboost_system.so.1.78.0           resources
libDeviceProperty.so             libQt6DesignerComponents.so.6  libQt6WebEngineWidgets.so.6                   libboost_thread.so.1.78.0           rules
libDevicePropertyProto.so        libQt6Gui.so.6                 libQt6Widgets.so.6                            libboost_timer.so.1.78.0            sqlite3
libEventSource.so                libQt6Help.so.6                libQt6XcbQpa.so.6                             libcrypto.so                        translations
libEventsView.so                 libQt6Multimedia.so.6          libQt6Xml.so.6                                libcrypto.so.1.1
libGenericHierarchy.so           libQt6MultimediaQuick.so.6     libQtPropertyBrowser.so                       libexec
libGpuInfo.so                    libQt6MultimediaWidgets.so.6   libQuiverContainers.so                        libexporter.so
libGpuTraits.so                  libQt6Network.so.6             libQuiverEvents.so                            libicudata.so.71
```



## 多版本管理

如果安装了多个nsys的版本，那么如果使用命令启动的时候默认的版本是可以选择的

```bash
(base) zyd@P7479785A244:/opt/nvidia/nsight-systems$ sudo update-alternatives --config nsys-ui
There are 3 choices for the alternative nsys-ui (providing /usr/local/bin/nsys-ui).

  Selection    Path                                                        Priority   Status
------------------------------------------------------------
* 0            /opt/nvidia/nsight-systems/2024.7.2/host-linux-x64/nsys-ui   0         auto mode
  1            /opt/nvidia/nsight-systems/2023.4.4/host-linux-x64/nsys-ui   0         manual mode
  2            /opt/nvidia/nsight-systems/2024.5.1/host-linux-x64/nsys-ui   0         manual mode
  3            /opt/nvidia/nsight-systems/2024.7.2/host-linux-x64/nsys-ui   0         manual mode
```

### 注册新版本

例如下面的操作：

是针对`nsys-ui`  和 ` nsys` 分别进行注册管理 

```bash
sudo update-alternatives --install /usr/local/bin/nsys-ui nsys-ui /opt/nvidia/nsight-systems/2024.7.2/host-linux-x64/nsys-ui 100
sudo update-alternatives --install /usr/local/bin/nsys nsys /opt/nvidia/nsight-systems/2024.7.2/host-linux-x64/nsys 100
```

### 删除不需要的版本

```bash
sudo update-alternatives --remove nsys-ui /opt/nvidia/nsight-systems/2024.7.2/host-linux-x64/nsys-ui
```

### 切换版本

```bash
sudo update-alternatives --config nsys-ui
```



#  CLI

nsisght systems提供了两种使用方式，一种是命令行 一种是GUI。在PC或者有图形界面的系统上我们可以使用GUI来方便的操作i，但是在一些没有图形界面的系统上我们只能使用CLI（命令行）来进行操作

## 典型的CLI命令

这里列举一些常用的CLI命令，（需要有root权限）

* 为了分析程序在GPU0 上的工作负载

  ```bash
  sudo nsys prof -t cuda,osrt,nvtx --gpu-metrics-device=0 -f true -o report_name <your_app>
  ```

* 为了分析在DLA上上的工作负载

  ```bash
  sudo nsys prof -t cuda,osrt,nvtx,cudla,tegra-accelerator --gpu-metirics-device=0 --accelerator-trace=tegra-accelerators -f true -o report-name <your_app>
  ```

* 为了分析在一段时间内，并且使用 `delay`(-y)和`duration`(-d)

  ```bash
  sudo nsys profile --delay=10 --duration=15 .....
  ```

* 使用API来控制分析的的范围，（-c） 使用cuda profile APIS 来确定分析程序的范围

  ```bash
  sudo nsys profile --capture-range=cudaProfilerApi ....
  ```

  这时需要对程序代码做一些修改

  * include cuda 头文件"cuda_profile_api.h"
  * 在代码中添加`cudaProfileStart()`和`cudaProfileStop()`在代码段的前后
  * 重新编译程序

* 为了收集backtrace on cuda apis ,需要设定一个模式，这会加大开销

  ```bash
  sudo nsys profile -t cuda,osrt,nvtx --cudabacktrace=all:100 -f true -o report_name <your_app>
  ```

* 可以分析多个进程，使用一个bash脚本来运行所有的进程，然后使用nsigth来分析这个脚本

  ```bash
  sudo nsys profile [options] ./your_script
  ```

  在QNX系统上需要添加一个参数`--process-scope=system-wide`

  ```bash
  sudo nsys profile [options] --process-scope=system-wide ./your_script
  ```

* 为了降低分析时的开销，可以禁止采样或者设置采样频率

  ```bash
  sudo nsys profile --sample=none --coutxse=none ...
  sudo nsys profile --sampling-frequency=8000 --osrt-threshold=10000 ....
  ```

#  系统级的程序调优

* 平衡在CPU和GPU上的工作负载
* 发现为使用的GPU和CPU时间
* 发现不必要的同步
* 发现优化时机
* 提高应用性能

# 在车载平台上使用

**在使用nvidia的drive orin时 我们需要使用与drive os匹配的版本，否则将会出现不可预知的错误。**

在安装Drive OS SDK的安装包时会安装对应的版本。

在HOST上，安装和Drive OS SDK一同发布的deb包

```bash
sudo dpkj -i NsightSystems-linux-nda-202*.*deb
sudo apt update
sudo apt install nsight-systems-202*
```

在QNX系统上，需要root权限的读写

```bash
mkqnx6fs /dev/vblk_ufs50
mount /dev/vblk_ufs50 /
mkdir /tmp /opt /opt/nvidia /root
```

##  在HOST上使用Nsight Systems GUI进行分析

在HOST远程分析我们在orin上运行的程序需要设置orin的ssh 

## 在设备上使用Nsight Systems CLI进行分析

我们也可以在orin上使用命令行的方式来在本地使用Nsight Systems

```bash
sudo nsys profile -y 10 -d 15 -w true -t "cuda,osrt,nvtx,cudla" -f true -o /path_to_save/report1 --accelerator-trace=tegra-accelerates <your_app>
```

* prifile:开始分析
* -y：延迟收集 单位秒
* -d：收集时长，单位秒
* -w true ：
* -t：设置需要跟踪的API
* -f ：如果时true 覆盖已经存在的outputfile
* -o：输出报告的文件名，会在结束收集时创建
* --accelerator-trace ：收集其他加速器的工作负载，对于tegra-accelerators 就是DLA

如果希望在生成报告时不kill程序，并且自己控制分析的时间。可以使用下面的方法

打开两个终端

```bash
sudo nsys launch --session-new=sessname ...<your_app>
```

在第二个终端中，如果你希望开始进行分析

```bash
sudo nsys start --sennsion=sessname -t cuda,osrt,nvtx --gpu-metrices-device=0 -f true -o report_name ...
```

当你想停止分析时在第二个终端中输入

```bash
sudo nsys stop --session=sessname
```

# profile告诉了我们什么

![Snipaste_2024-07-28_22-38-28](Nsight System经验/Snipaste_2024-07-28_22-38-28.png)

![image-20240728225223577](Nsight System经验/image-20240728225223577.png)



# nsys不能抓取到数据

有的时候nsys抓取数据，查看rep文件根本没有对应的cuda信息。那么可以查看一下是不是 /tmp 文件夹没有写入权限或者 太小了。

https://docs.nvidia.com/nsight-systems/UserGuide/index.html

By default, Nsight Systems writes temporary files to `/tmp` directory. If you are using a system that does not allow writing to `/tmp` or where the `/tmp` directory has limited storage you can use the TMPDIR environment variable to set a different location. An example:

```
TMPDIR=/testdata ./bin/nsys profile -t cuda matrixMul
```

如果是需要改写 temporary files路径，那么推荐写入到  filesystem为 tmpfs 的路径下。提高性能（临时写入）。

mpfs是什么呢? 其实是一个临时文件系统，驻留在内存中，所以/dev/shm/这个目录不在硬盘上，而是在内存里。因为是在内存里，所以读写非常快，可以提供较高的访问速度。