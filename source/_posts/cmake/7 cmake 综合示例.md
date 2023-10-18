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

### 2.1 hello.cu

```cuda
#include "../common/common.h"
#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    CHECK(cudaDeviceReset());
    return 0;
}



```

### 2.2 build.sh

编译使用的是脚本`build.sh`因此先看一下这个文件

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

### 2.3 CMakeLists.txt

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

### 2.4 arm_qnx_toolchain.cmake

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

### 2.5 x86_toolchain.cmake

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

## 4 cuda和c++混合编译

修改上面的hello.cu，分为一个main.cpp和hello.cu 。在一个文件夹内。cmake文件都不需要修改，只修改源码部分如下

```shell
.
├── arm_qnx_toolchain.cmake
├── Build
├── build.sh
├── CMakeLists.txt
├── hello.cu
├── main.cpp
├── Makefile_back
└── x86_toolchain.cmake

1 directory, 7 files

```

main.cpp

```C
#include <iostream>

void helloCPU() {
    std::cout << "Hello from CPU!" << std::endl;
}

extern void helloFromGPU();

int main() {
    helloCPU();
    helloFromGPU();
    return 0;
}
```

hello.cu

```cuda
#include <iostream>

__global__ void helloGPU() {
    printf("Hello from GPU!\n");
}

void helloFromGPU() {
    helloGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
}
```



## 5 与GPT的对话

cmake的值令比较多，但是之间的相互关系没弄太明白。还是问了GPT才有一个比较直观的了解。

### 编写一个cuda 的cmake文件

2023/8/25 13:47:01

当编写一个 CUDA 的 CMake 文件时，我们需要确保正确配置以下内容：

1. 检查 CUDA 是否可用；
2. 设置工程名称和最低要求的 CMake 版本；
3. 添加源文件、头文件和其他依赖项；
4. 配置编译选项和链接选项；
5. 生成目标文件。

下面是一个示例 CUDA CMake 文件的基本结构：

```
cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MyCudaProject)

# 检查 CUDA 是否可用
find_package(CUDA REQUIRED)

# 设置 CUDA 架构类型（可根据需要更改）
set(CUDA_ARCHITECTURES "60")

# 添加源文件和头文件
set(SOURCES
    main.cu
)
set(HEADERS
    kernel.h
)

# 配置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_STANDARD 14)

# 添加其他依赖项（如果有）
# find_package(<package_name> REQUIRED)

# 生成目标文件
add_executable(MyCudaExecutable ${SOURCES} ${HEADERS})

# 链接 CUDA 库和其他依赖项（如果有）
target_link_libraries(MyCudaExecutable CUDA::cudart)
```

请注意，在以上示例中，我们使用了 `find_package(CUDA REQUIRED)` 来检查 CUDA 是否可用。然而，自 CUDA 11.3 起，NVIDIA 建议使用新的 `find_package(CUDAToolkit REQUIRED)` 来代替旧的 `find_package(CUDA REQUIRED)`。

此外，还需要根据您的项目需求进行适当的修改。您可以在 `SOURCES` 和 `HEADERS` 中添加或删除源文件和头文件，并在 `CMAKE_CUDA_FLAGS` 中配置适当的编译选项。

最后，运行 CMake 生成您的构建系统所需的 Makefile 或 Visual Studio 工程文件等。



2023/8/25 13:47:18

### 修改为交叉编译的版本



2023/8/25 13:47:35

若要使用 CMake 进行 CUDA 的交叉编译，您需要提供交叉编译工具链，并根据目标平台的架构和配置进行相应的设置。下面是一个示例 CUDA 交叉编译的 CMake 文件：

```
cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MyCudaProject)

# 设置交叉编译工具链
set(CMAKE_C_COMPILER /path/to/cross-compiler/bin/gcc)
set(CMAKE_CXX_COMPILER /path/to/cross-compiler/bin/g++)

# 检查 CUDA 是否可用
find_package(CUDA REQUIRED)

# 设置 CUDA 架构类型（可根据需要更改）
set(CUDA_ARCHITECTURES "60")

# 添加源文件和头文件
set(SOURCES
    main.cu
)
set(HEADERS
    kernel.h
)

# 配置编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=${CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_STANDARD 14)

# 添加其他依赖项（如果有）
# find_package(<package_name> REQUIRED)

# 生成目标文件
add_executable(MyCudaExecutable ${SOURCES} ${HEADERS})

# 链接 CUDA 库和其他依赖项（如果有）
target_link_libraries(MyCudaExecutable CUDA::cudart)
```

在上述示例中，您需要将 `/path/to/cross-compiler` 替换为实际的交叉编译工具链路径，以确保 CMake 使用正确的交叉编译器。

请记住，交叉编译可能需要更多的配置和调整，具体取决于您的目标平台和环境。因此，以上示例仅供参考，您需要根据自己的需求进行相应的修改。

同样，在生成构建系统时，请使用适当的交叉编译命令和参数。例如，使用 `cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake ..` 来指定交叉编译工具链文件。

2023/8/25 13:48:21

### 什么时候需要设定 CMAKE_CUDA_HOST_COMPILER 参数



2023/8/25 13:48:45

在某些情况下，您可能需要设置 `CMAKE_CUDA_HOST_COMPILER` 参数。这个参数用于指定在主机上编译 CUDA 源文件时要使用的编译器。

通常，如果您正在进行交叉编译或使用不同于默认值的编译器，则需要设置 `CMAKE_CUDA_HOST_COMPILER`。下面是一些常见的情况：

1. 交叉编译：当您在一个平台上开发 CUDA 代码，但希望将其运行在另一个平台时，您需要使用交叉编译工具链，并设置 `CMAKE_CUDA_HOST_COMPILER` 来指定正确的主机编译器。这样可以确保在主机上编译 CUDA 源文件时使用交叉编译器而不是本地编译器。
2. 特定编译器要求：CUDA 可能与特定版本的编译器相关联。如果您希望使用不同于默认主机编译器的特定编译器版本，您可以通过设置 `CMAKE_CUDA_HOST_COMPILER` 参数来指定所需的编译器。

以下是一个示例，展示了如何设置 `CMAKE_CUDA_HOST_COMPILER` 参数：

```
cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MyCudaProject)

# 设置交叉编译工具链和主机编译器
set(CMAKE_C_COMPILER /path/to/cross-compiler/bin/gcc)
set(CMAKE_CXX_COMPILER /path/to/cross-compiler/bin/g++)
set(CMAKE_CUDA_HOST_COMPILER /path/to/cross-compiler/bin/gcc)

# 检查 CUDA 是否可用
find_package(CUDA REQUIRED)

# 其他配置和生成目标文件的步骤...
```

在上述示例中，我们设置了 `CMAKE_CUDA_HOST_COMPILER` 参数为交叉编译工具链中的 `gcc` 编译器路径。这样，在主机上编译 CUDA 源文件时，CMake 将使用指定的编译器。

请注意，是否需要设置 `CMAKE_CUDA_HOST_COMPILER` 取决于您的具体需求和环境。大多数情况下，默认的主机编译器即可正常工作，无需手动设置该参数。



2023/8/25 13:50:40

### 不使用 find_package

不使用 find_package 重新编写一个交叉编译 cuda 的toolchain.cmake 和对应的cmakelist文件

2023/8/25 13:52:20

当您不使用 `find_package` 命令时，可以通过编写自定义的工具链文件 `toolchain.cmake` 和相应的 CMakeLists.txt 文件来进行交叉编译 CUDA。

首先，创建一个名为 `toolchain.cmake` 的文件，并在其中定义交叉编译所需的工具链路径和选项。以下是一个示例 `toolchain.cmake` 文件的结构：

```
cmake
# 设置交叉编译工具链
set(CMAKE_SYSTEM_NAME Linux)  # 目标系统名称
set(CMAKE_C_COMPILER /path/to/cross-compiler/bin/gcc)  # C 编译器路径
set(CMAKE_CXX_COMPILER /path/to/cross-compiler/bin/g++)  # C++ 编译器路径

# 设置 CUDA 相关变量
set(CUDA_TOOLKIT_ROOT_DIR /path/to/cuda/toolkit)  # CUDA 工具包路径
set(CMAKE_CUDA_HOST_COMPILER /path/to/cross-compiler/bin/cpp)  # 主机编译器路径
set(CMAKE_CUDA_ARCHITECTURES "60")  # CUDA 架构类型

# 其他交叉编译选项和配置...
```

请根据您的实际情况修改上述文件中的路径和选项，确保它们与您的交叉编译环境一致。

接下来，创建一个名为 `CMakeLists.txt` 的文件，用于设置项目的源文件、目标文件和其他构建相关的信息。以下是一个示例 `CMakeLists.txt` 文件的基本结构：

```
cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MyCudaProject)

# 加载交叉编译工具链
set(CMAKE_TOOLCHAIN_FILE /path/to/toolchain.cmake)

# 添加源文件和头文件
set(SOURCES
    main.cu
)
set(HEADERS
    kernel.h
)

# 配置编译选项
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")  # 示例：添加其他编译选项

# 添加其他依赖项（如果有）
# find_package(<package_name> REQUIRED)

# 生成目标文件
add_executable(MyCudaExecutable ${SOURCES} ${HEADERS})

# 链接 CUDA 库和其他依赖项（如果有）
target_link_libraries(MyCudaExecutable CUDA::cudart)
```

在上述示例中，我们通过 `set(CMAKE_TOOLCHAIN_FILE /path/to/toolchain.cmake)` 指定了自定义的工具链文件路径。

请注意，示例中的路径 `/path/to/toolchain.cmake` 需要替换为您实际的 `toolchain.cmake` 文件路径。

此外，您还需要根据您的项目需求进行适当的修改。您可以在 `SOURCES` 和 `HEADERS` 中添加或删除源文件和头文件，并在 `CMAKE_CUDA_FLAGS` 中配置适当的编译选项。

最后，运行 CMake 生成您的构建系统所需的 Makefile 或 Visual Studio 工程文件等。使用以下命令运行 CMake：

```
bash
cmake /path/to/source/directory
```

确保将 `/path/to/source/directory` 替换为包含 `CMakeLists.txt` 文件的实际源代码目录路径。



2023/8/25 13:53:04

### 添加一些 其他交叉编译选项和配置 的配置



2023/8/25 13:53:53

当进行交叉编译 CUDA 时，您可能需要添加一些其他的交叉编译选项和配置。这些选项和配置将根据您的目标平台和项目需求而有所不同。以下是一个示例 `toolchain.cmake` 文件和对应的 `CMakeLists.txt` 文件，演示了如何添加其他交叉编译选项和配置：

**toolchain.cmake:**

```
cmake
# 设置交叉编译工具链
set(CMAKE_SYSTEM_NAME Linux)  # 目标系统名称
set(CMAKE_C_COMPILER /path/to/cross-compiler/bin/gcc)  # C 编译器路径
set(CMAKE_CXX_COMPILER /path/to/cross-compiler/bin/g++)  # C++ 编译器路径

# 设置 CUDA 相关变量
set(CUDA_TOOLKIT_ROOT_DIR /path/to/cuda/toolkit)  # CUDA 工具包路径
set(CMAKE_CUDA_HOST_COMPILER /path/to/cross-compiler/bin/cpp)  # 主机编译器路径
set(CMAKE_CUDA_ARCHITECTURES "60")  # CUDA 架构类型

# 其他交叉编译选项和配置
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")  # 静态链接标准库
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")  # 禁用共享库链接
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")  # 禁用共享库链接
```

**CMakeLists.txt:**

```
cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MyCudaProject)

# 加载交叉编译工具链
set(CMAKE_TOOLCHAIN_FILE /path/to/toolchain.cmake)

# 添加源文件和头文件
set(SOURCES
    main.cu
)
set(HEADERS
    kernel.h
)

# 配置编译选项
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")  # 示例：添加其他编译选项

# 添加其他依赖项（如果有）
# find_package(<package_name> REQUIRED)

# 生成目标文件
add_executable(MyCudaExecutable ${SOURCES} ${HEADERS})

# 链接 CUDA 库和其他依赖项（如果有）
target_link_libraries(MyCudaExecutable CUDA::cudart)
```

在上述示例中，我们通过在 `toolchain.cmake` 文件中设置以下内容来添加其他交叉编译选项和配置：

- `CMAKE_EXE_LINKER_FLAGS`：用于设置链接器的选项。在示例中，我们使用 `-static-libgcc -static-libstdc++` 来静态链接 C 和 C++ 的标准库。
- `CMAKE_SHARED_LIBRARY_LINK_C_FLAGS` 和 `CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS`：用于禁用共享库的链接。这样可以确保只生成可执行文件而不是共享库。

您可以根据自己的需求修改这些选项和配置。请注意，具体的选项和配置可能因交叉编译平台和环境而异。因此，上述示例仅供参考，您需要根据自己的情况进行相应的修改。

### find_package 和enable_language 用哪一个？

2023/8/25 14:00:36

`find_package` 和 `enable_language` 是用于不同目的的 CMake 命令，具体取决于您的需求。

- `find_package` 用于查找和配置外部软件包或库。它通过在系统上搜索指定软件包的头文件、库文件或配置文件来确定软件包是否可用，并据此设置相关的变量和选项。这通常用于自动检测和配置项目所需的依赖项。例如，使用 `find_package(CUDA REQUIRED)` 可以查找 CUDA 并将其配置为项目的依赖项。
- `enable_language` 用于启用特定的编程语言支持。它告诉 CMake 在构建过程中启用指定的编程语言。默认情况下，CMake 已经启用了常见的编程语言，如 C 和 C++。但对于某些较新或非常见的编程语言，可能需要显式地使用 `enable_language` 命令来启用它们。例如，使用 `enable_language(CUDA)` 可以启用 CUDA 编程语言的支持。

因此，这两个命令有着不同的作用和用途。如果您需要查找和配置外部软件包或库，以满足项目的依赖项，那么使用 `find_package` 是合适的选择。而如果您想要启用特定的编程语言支持，例如 CUDA，那么使用 `enable_language` 是更合适的

### 交叉编译时 CMAKE_CUDA_HOST_COMPILER什么时候设定为gcc 什么时候设定为nvcc

2023/8/25 14:35:39

在进行交叉编译时，`CMAKE_CUDA_HOST_COMPILER`的设置取决于您要使用的工具链和目标平台。

当您希望使用GCC作为主机编译器时，将`CMAKE_CUDA_HOST_COMPILER`设置为GCC的路径。这通常在以下两种情况下发生：

1. 当您希望将CUDA代码与C/C++代码混合编译，并且主机端的代码由GCC编译器处理时，可以将`CMAKE_CUDA_HOST_COMPILER`设置为GCC的路径。
2. 当您在交叉编译环境中使用GCC作为主机编译器时，例如在ARM架构上进行交叉编译，您需要将`CMAKE_CUDA_HOST_COMPILER`设置为适用于该交叉编译环境的GCC路径。

相反地，当您希望将NVCC作为主机编译器时，将`CMAKE_CUDA_HOST_COMPILER`设置为nvcc的路径。这通常在以下情况下发生：

1. 当您只使用CUDA编程模型而没有主机端的C/C++代码时，您可以将`CMAKE_CUDA_HOST_COMPILER`设置为nvcc的路径。

总之，`CMAKE_CUDA_HOST_COMPILER`的设置取决于您的需求和所使用的工具链。

### 关于 CMAKE_CUDA_FLAGS

`CMAKE_CUDA_FLAGS` 用于设置 NVIDIA CUDA 编译器（通常是 `nvcc`）的编译选项，它会影响 `nvcc` 命令的行为。这些选项被传递给 `nvcc` 编译器，以控制 CUDA 源代码的编译和链接行为。详细编译选项参考[NVCC Command Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options)

`CMAKE_CUDA_FLAGS` 变量用于设置 NVIDIA CUDA 编译器 `nvcc` 的编译选项。以下是一些常见选项及其含义，以及一个示例：

1. **-arch=sm_xx**：这个选项指定目标 GPU 的架构版本，其中 `xx` 是特定架构的版本号。例如，`-arch=sm_61` 表示目标 GPU 架构是 Compute Capability 6.1。
2. **-O[0-3]**：设置优化级别。`-O0` 表示没有优化，`-O3` 表示最高级别的优化。
3. **-g**：生成调试信息，以便使用调试器进行 CUDA 代码的调试。
4. **-G**：生成 GPU 代码的调试信息，用于在 GPU 代码中进行调试。
5. **-Xcompiler**：用于传递额外的编译器选项，这些选项将传递给主机 C/C++ 编译器。
6. **-Xptxas**：传递给 ptxas 汇编器的选项，可以影响 PTX 代码生成。
7. **-Xlinker**：用于传递给链接器的选项，用于配置链接过程。
8. **-lineinfo**：生成 GPU 代码中的行号信息，以用于调试。
9. **-Xcudafe**：传递给 cudafe（CUDA 前端编译器）的选项，可以影响 CUDA 源代码的处理。
10. **-rdc=true/false**：启用或禁用 CUDA 动态并行性。设置为 `true` 允许动态并行性，`false` 禁用它。
11. **-Werror**：将编译警告视为错误，强制开发人员修复所有警告。
12. **-lineinfo**： 为设备代码生成行号信息。在使用Nsight Compute时非常有用
13. **-default-stream**：`--default-stream` {`legacy`|`null`|`per-thread`} (`per-thread`) 。比较常用`--default-stream per-thread `

示例 `CMAKE_CUDA_FLAGS` 设置：

```
cmake
# 设置 CUDA 架构为 Compute Capability 7.0
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")

# 启用最高级别优化
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# 生成调试信息
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")

# 将额外编译器选项传递给主机编译器
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

# 生成 GPU 代码的调试信息
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

# 传递给链接器的选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker -L/path/to/custom/lib")

# 生成 GPU 代码中的行号信息
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# 传递给 cudafe 的选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

# 启用 CUDA 动态并行性
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

# 将编译警告视为错误
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror")

# 设置编译选项 源代码反汇编视图用于在源代码和汇编指令级别显示内核的分析结果。为了能够查看内核源代码，您需要使用-lineinfo选项编译代码。如果不使用此编译器选项，则只显示反汇编视图。
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g --default-stream per-thread")
```



# 附录：

* [NVCC Command Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-command-options)
