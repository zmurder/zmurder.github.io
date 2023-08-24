# 7 cmake综合示例

比较简单的例子在“2 cmake简单示例”中介绍过了。这里使用一个综合的例子实现以下功能：

1. cmake实现交叉编译和x86的编译（脚本传参实现区分）
2. 使用参数“[`-DCMAKE_TOOLCHAIN_FILE`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-D)”实现工具链的配置
3. cmake实现交叉编译和x86编译cuda程序

## 1 文件结构

具体的文件结构如下

```shell
zyd@zyd:~/WorkSpace/zyd/note/cuda/code/chapter01$ tree -L 1
.
├── arm_qnx_toolchain.cmake
├── Build
├── build.sh
├── CMakeLists.txt
├── hello.cu
├── Makefile_back
└── x86_toolchain.cmake

1 directory, 7 files
```

* Build：cmake编译生成文件目录
* build.sh：编译脚本 可以传入参数实现不同平台的编译
* CMakeLists.txt：cmake命令依赖的文件
* hello.cu：cuda源码文件
*  Makefile_back：之前的makfile文件，先在用cmake了，没什么用
* arm_qnx_toolchain.cmake和x86_toolchain.cmake分别是用于交叉编译arm qnx和编译x86的工具链配置文件

## 2 文件内容

编译使用的是脚本`build.sh`因此先看一下这个文件

### 2.1 build.sh

```shell
#!/bin/bash

#判断Build文件夹是否存在，不存在就创建。编译的程序就在这里
if [ ! -d "Build" ];then
    mkdir Build
else
    echo "Build dir is exist"
fi

cd Build

#这里主要是设置了一些环境变量，交叉编译使用。具体使用参看arm_qnx_toolchain.cmake文件
if [ "$1" != "x86" ] &&  [ "$1" != "X86" ]; then
	#设置交叉编译的环境变量 否则会出现"The C compiler identification is unknown"的错误
	QNX_BASE=/media/zyd/VisionPerception40Env/toolChainsQnx710
	QNX_HOST=${QNX_BASE}/host/linux/x86_64
	QNX_TARGET=${QNX_BASE}/target/qnx7
	CUDA_DIR=${QNX_BASE}/cudaSafe11.4

	PATH=${QNX_HOST}/usr/bin:${PATH}
	# PATH=${CUDA_DIR}/bin:${PATH}

	export PATH QNX_BASE QNX_HOST QNX_TARGET CUDA_DIR
fi

# 不同的参数配置不同的工具链 -DCMAKE_TOOLCHAIN_FILE参数指定
if [ "$1" = "x86" ] ||  [ "$2" = "x86" ]; then
	cmake -DARCH_X86=1 -DCMAKE_TOOLCHAIN_FILE=../x86_toolchain.cmake .. && make
else
	cmake -DCMAKE_TOOLCHAIN_FILE=../arm_qnx_toolchain.cmake .. && make
fi

```

上面脚本在Build目录下调用了`cmake ..`，因此下一步查看文件`CMakeLists.txt`

### 2.2 CMakeLists.txt

```cmake
# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.16)

# 打印编译的具体过程信息
# set(CMAKE_VERBOSE_MAKEFILE ON) 

# 项目信息
project (hello)
# project (hello LANGUAGES CUDA CXX)

# 功能类似与project (hello LANGUAGES CUDA)
# Enables support for the named languages in CMake
enable_language(CUDA)

# 指定头文件目录 PROJECT_SOURCE_DIR: 当前工程的源码路径
include_directories(${PROJECT_SOURCE_DIR})
include_directories(../common)


# include_directories($ENV{QNX_BASE}/target/qnx7/usr/include)
# include_directories($ENV{CUDA_DIR}/targets/aarch64-qnx/include)
# include_directories(/media/zyd/VisionPerception40Env/toolChainsQnx710/target/qnx7/usr/include)

# link_directories($ENV{CUDA_DIR}/target/qnx7/usr/lib)
# link_directories($ENV{CUDA_DIR}/targets/aarch64-qnx/lib)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(./ DIR_SRCS)
# message("DIR_SRCS is $ENV{DIR_SRCS}")
# 添加 src 子目录
# add_subdirectory(src)

# 指定生成目标
if (DEFINED ARCH_X86)
	set(PROJECT_NAME "helloX86")
else()
	set(PROJECT_NAME "hello")
endif ()

add_executable(${PROJECT_NAME} ${DIR_SRCS})

# 添加链接库
# target_link_libraries(hello fun)

```

由于需要编译cuda程序，因此需要设定`enable_language(CUDA)`，这个也可以使用`project (hello LANGUAGES CUDA)`来替代。

因为需要编译为两个平台，一个交叉编译一个本地PC编译，在`build.sh`中指定了cmake参数`-DCMAKE_TOOLCHAIN_FILE`来配置工具链。下面就看一下两个工具链的配置文件arm_qnx_toolchain.cmake和x86_toolchain.cmake

### 2.3 arm_qnx_toolchain.cmake

先看一下交叉编译工具链的配置，参考[cmake-toolchains(7)](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html#id8)

```cmake
# 要为其生成CMake的操作系统的名称
set(CMAKE_SYSTEM_NAME QNX)
# 目标体系结构的CMake标识符。
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 配置交叉编译工具链gcc和g++
set(QNX_HOST "$ENV{QNX_HOST}")
set(CMAKE_C_COMPILER ${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.1.0-gcc)
set(CMAKE_CXX_COMPILER ${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++)

set(CMAKE_CUDA_STANDARD 11)

# 配置cuda的编译工具链nvcc
set(CMAKE_CUDA_COMPILER "$ENV{QNX_BASE}/cudaSafe11.4/bin/nvcc")

# cuda的目标架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

# set(CMAKE_SYSROOT $ENV{QNX_TARGET})

# CMAKE_CUDA_HOST_COMPILER会选择编译CUDA语言文件的主机代码时要使用的编译器可执行文件。这映射到nvcc-ccbin选项
set(CMAKE_CUDA_HOST_COMPILER "${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++")

```

比较关键的几个点

* `CMAKE_C_COMPILER`和`CMAKE_CXX_COMPILER`是设置交叉编译工具链gcc和g++的路经

* `CMAKE_CUDA_COMPILER`配置`.cu`GPU的编译工具链`nvcc`的路经

* `CMAKE_CUDA_STANDARD`设定CUDA标准

* `CMAKE_CUDA_ARCHITECTURES`设定GPU的架构

* `CMAKE_CUDA_HOST_COMPILER`需要设定为交叉编译工具g++的路经。如果不设定这一个会出现以下的类似错误

  ```shell
  zyd@zyd:~/WorkSpace/zyd/note/cuda/code/chapter01$ ./build.sh
  -- The C compiler identification is GNU 8.3.0
  -- The CXX compiler identification is GNU 8.3.0
  -- Check for working C compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.1.0-gcc
  -- Check for working C compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.1.0-gcc -- works
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Check for working CXX compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++
  -- Check for working CXX compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++ -- works
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- The CUDA compiler identification is unknown
  -- Check for working CUDA compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/cudaSafe11.4/bin/nvcc
  -- Check for working CUDA compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/cudaSafe11.4/bin/nvcc -- broken
  CMake Error at /usr/local/share/cmake-3.16/Modules/CMakeTestCUDACompiler.cmake:46 (message):
    The CUDA compiler
  
      "/media/zyd/VisionPerception40Env/toolChainsQnx710/cudaSafe11.4/bin/nvcc"
  
    is not able to compile a simple test program.
  
    It fails with the following output:
  
      Change Dir: /home/zyd/WorkSpace/zyd/note/cuda/code/chapter01/Build/CMakeFiles/CMakeTmp
      
      Run Build Command(s):/media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/make cmTC_612bd/fast && /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/make -f CMakeFiles/cmTC_612bd.dir/build.make CMakeFiles/cmTC_612bd.dir/build
      make[1]: Entering directory '/home/zyd/WorkSpace/zyd/note/cuda/code/chapter01/Build/CMakeFiles/CMakeTmp'
      Building CUDA object CMakeFiles/cmTC_612bd.dir/main.cu.o
      /media/zyd/VisionPerception40Env/toolChainsQnx710/cudaSafe11.4/bin/nvcc     -x cu -c /home/zyd/WorkSpace/zyd/note/cuda/code/chapter01/Build/CMakeFiles/CMakeTmp/main.cu -o CMakeFiles/cmTC_612bd.dir/main.cu.o
      cc1plus: fatal error: cuda_runtime.h: No such file or directory
      compilation terminated.
      CMakeFiles/cmTC_612bd.dir/build.make:65: recipe for target 'CMakeFiles/cmTC_612bd.dir/main.cu.o' failed
      make[1]: *** [CMakeFiles/cmTC_612bd.dir/main.cu.o] Error 1
      make[1]: Leaving directory '/home/zyd/WorkSpace/zyd/note/cuda/code/chapter01/Build/CMakeFiles/CMakeTmp'
      Makefile:121: recipe for target 'cmTC_612bd/fast' failed
      make: *** [cmTC_612bd/fast] Error 2
      
      
  
    
  
    CMake will not be able to correctly generate this project.
  Call Stack (most recent call first):
    CMakeLists.txt:13 (enable_language)
  
  
  -- Configuring incomplete, errors occurred!
  See also "/home/zyd/WorkSpace/zyd/note/cuda/code/chapter01/Build/CMakeFiles/CMakeOutput.log".
  See also "/home/zyd/WorkSpace/zyd/note/cuda/code/chapter01/Build/CMakeFiles/CMakeError.log".
  
  ```

### 2.3 x86_toolchain.cmake

这个相对与交叉编译工具链的比较简单

```cmake
set(CMAKE_CUDA_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

```

## 3 编译

交叉编译

```shell
zyd@zyd:~/WorkSpace/zyd/note/cuda/code/chapter01$ ./build.sh
-- The C compiler identification is GNU 8.3.0
-- The CXX compiler identification is GNU 8.3.0
-- Check for working C compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.1.0-gcc
-- Check for working C compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.1.0-gcc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++
-- Check for working CXX compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The CUDA compiler identification is NVIDIA 11.4.207
-- Check for working CUDA compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/cudaSafe11.4/bin/nvcc
-- Check for working CUDA compiler: /media/zyd/VisionPerception40Env/toolChainsQnx710/cudaSafe11.4/bin/nvcc -- works
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Configuring done
-- Generating done
-- Build files have been written to: /home/zyd/WorkSpace/zyd/note/cuda/code/chapter01/Build
Scanning dependencies of target hello
[ 50%] Building CUDA object CMakeFiles/hello.dir/hello.cu.o
[100%] Linking CUDA executable hello
[100%] Built target hello
```

编译PC版版本，（切换版本编译需要删除Build目录内容）

```shell
zyd@zyd:~/WorkSpace/zyd/note/cuda/code/chapter01$ ./build.sh x86
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- The CUDA compiler identification is NVIDIA 11.8.89
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc -- works
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Configuring done
-- Generating done
CMake Warning:
  Manually-specified variables were not used by the project:

    ARCH_X86


-- Build files have been written to: /home/zyd/WorkSpace/zyd/note/cuda/code/chapter01/Build
Scanning dependencies of target helloX86
[ 50%] Building CUDA object CMakeFiles/helloX86.dir/hello.cu.o
[100%] Linking CUDA executable helloX86
[100%] Built target helloX86
```

