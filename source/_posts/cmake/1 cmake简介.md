# 1 cmake简介

cmake允许开发者编写一种平台无关的 CMakeList.txt 文件来定制整个编译流程，然后再根据目标用户的平台进一步生成所需的本地化  Makefile 和工程文件，如 Unix 的 Makefile 或 Windows 的 Visual Studio 工程。从而做到“Write once, run everywhere”。

其他的make工具如GNU Make ，QT 的 qmake ，微软的 MSnmake，BSD Make（pmake），Makepp，等等。这些 Make 工具遵循着不同的规范和标准，所执行的 Makefile 格式也千差万别。

## 1.1 编写流程

在linux下使用cmake生成makefile并编译的流程如下：

* 编写cmake的配置文件CMakeLists.txt
* 执行命令 cmake PATH 或者 ccmake PATH 生成 Makefile。其中， PATH 是 CMakeLists.txt 所在的目录。
* 使用 make 命令进行

## 1.2 编译和源代码分离

* CMake 背后的逻辑思想是编译和源代码分离的原则。 
* 通常 CMakeLists.txt 是和源代码放在一起的。一般每个子目录下都有一个 CMakeLists.txt 用于组织该目录下的文件。 
* 子目录的 CMakeLists.txt 自动继承了父目录里的 CMakeLists.txt 所定义的一切宏、变量。这极大地减少了重复的代码。 
* 而针对具体的平台和配置，我们可以单独创建一个目录，然后在该目录下生成特定平台和配置的工程文件。这样能够做到具体的工程文件不会和源代码文件混搭在一起。 例如后面讲到的交叉编译的例子，会有一个指定工具链等的cmake文件

## 1.3 安装cmake

在官网https://cmake.org/download/下载安转包

```shell
#tar -xvf cmake-3.22.0-rc2.tar.gz        
#cd cmake-3.22.0-rc2
#./bootstrap
#make
#make install
 
cmake 会默认安装在 /usr/local/bin 下面
```

安装完成后可以查看cmake版本信息

```shell
zhaoyda~$ cmake --version
cmake version 3.16.3

CMake suite maintained and supported by Kitware (kitware.com/cmake).

```

