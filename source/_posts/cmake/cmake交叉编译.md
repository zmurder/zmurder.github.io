# 简介

一直对cmake的交叉编译有点晕晕的，这里不定期的总结一些交叉编译的内容吧。

# cmake交叉编译的例子

这里先说一下目录的结构

```bash
CMakeLists.txt            sumMatrix_pinned.cu           sumMatrix_unified.cu
arm_orin_toolchain.cmake  sumMatrix_pinned_nocopy.cu    sumMatrix_zerocopy.cu
build.sh*                 sumMatrix_register.cu         x86_toolchain.cmake
sumMatrix.cu              sumMatrix_register_nocopy.cu

```

这个是我的测试代码目录，每一个cu文件都需要编译为一个对应的可执行程序。

我的需求是可以使用cmake控制到底编译为x86版本的程序还是编译为arm版本的程序。

与cmake相关的是`CMakeLists.txt build.sh  x86_toolchain.cmake  arm_orin_toolchain.cmake`  

使用build.sh 来控制使用 `x86_toolchain.cmake 还是 arm_orin_toolchain.cmake` 来确定编译平台相关内容，`CMakeLists.txt`确定具体编译的代码。

可以参考https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html



build.sh 如下

```bash
#!/bin/bash

if [ ! -d "Build" ];then
    mkdir Build
else
    echo "Build dir is exist"
fi

cd Build

# if [ "$1" != "x86" ] &&  [ "$1" != "X86" ]; then
# 	#设置交叉编译的环境变量 否则会出现"The C compiler identification is unknown"的错误
# 	# ORIN_BASE=/media/zyd/VisionPerception40Env/toolChainsQnx710
# 	# ORIN_HOST=${ORIN_BASE}/host/linux/x86_64
# 	# ORIN_TARGET=${ORIN_BASE}/target/qnx7
# 	# CUDA_DIR=${ORIN_BASE}/cudaSafe11.4
# 	CUDA_DIR=/home/zyd/drive/drive-linux/filesystem/targetfs/usr/local/cuda-11.4

# 	# PATH=${ORIN_HOST}/usr/bin:${PATH}
# 	# PATH=${CUDA_DIR}/bin:${PATH}
# else
# 	CUDA_DIR=/usr/local/cuda
# fi

# export PATH ORIN_BASE ORIN_HOST ORIN_TARGET CUDA_DIR
# echo "ORIN_BASE=$ORIN_BASE"
# echo "ORIN_HOST=$ORIN_HOST"
# echo "ORIN_TARGET=$ORIN_TARGET"
# echo "CUDA_DIR=$CUDA_DIR"


# echo "PATH = ${PATH}"
if [ "$1" = "x86" ] ||  [ "$2" = "x86" ]; then
	cmake -DARCH_X86=1 -DCMAKE_TOOLCHAIN_FILE=../x86_toolchain.cmake .. && make
else
	cmake  -DCMAKE_TOOLCHAIN_FILE=../arm_orin_toolchain.cmake .. && make
fi

```



CMakeLists.txt如下：

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

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall")

# 指定头文件目录 PROJECT_SOURCE_DIR: 当前工程的源码路径
include_directories(../common)
include_directories(${PROJECT_SOURCE_DIR})
# include_directories($ENV{CUDA_DIR}/include)

# message("TTTTT $ENV{QNX_BASE}/target/qnx7/usr/include")
# include_directories($ENV{CUDA_DIR}/targets/aarch64-qnx/include)
# include_directories(/media/zyd/VisionPerception40Env/toolChainsQnx710/target/qnx7/usr/include)

# link_directories($ENV{CUDA_DIR}/target/qnx7/usr/lib)
# link_directories($ENV{CUDA_DIR}/targets/aarch64-qnx/lib)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
# aux_source_directory(./ DIR_SRCS)
set(DIR_SRCS sumMatrix.cu)

# message("DIR_SRCS is $ENV{DIR_SRCS}")
# 添加 src 子目录
# add_subdirectory(src)
message("ARCH_X86 = ${ARCH_X86}")

# 指定生成目标
if (ARCH_X86)
	set(PROJECT_NAME "sumMatrixX86")
	message("PROJECT_NAME = ${PROJECT_NAME}")
else()
	set(PROJECT_NAME "sumMatrix")
	message("PROJECT_NAME = ${PROJECT_NAME}")
endif ()

add_executable(${PROJECT_NAME} ${DIR_SRCS})
add_executable(sumMatrix_register sumMatrix_register.cu)
add_executable(sumMatrix_register_nocopy sumMatrix_register_nocopy.cu)
add_executable(sumMatrix_pinned sumMatrix_pinned.cu)
add_executable(sumMatrix_pinned_nocopy sumMatrix_pinned_nocopy.cu)
add_executable(sumMatrix_zerocopy sumMatrix_zerocopy.cu)
add_executable(sumMatrix_unified sumMatrix_unified.cu)



# 添加链接库
# target_link_libraries(hello fun)

```



arm_orin_toolchain.cmake 如下

```cmake
# 要为其生成CMake的操作系统的名称
set(CMAKE_SYSTEM_NAME Linux)
# 目标体系结构的CMake标识符。
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(TOOL_CHAINS "/opt/nvidia/driveos/toolchains")
# set(CMAKE_SYSROOT /home/zyd/drive/drive-linux/filesystem/targetfs)

# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
# set(CMAKE_STAGING_PREFIX /home/devel/stage)

# include_directories(/media/zyd/VisionPerception40Env/toolChainsQnx710/target/qnx7/usr/include)
include_directories(/home/zyd/drive/drive-linux/filesystem/targetfs/usr/include)

# 配置交叉编译工具链gcc和g++

set(CMAKE_C_COMPILER ${TOOL_CHAINS}/aarch64--glibc--stable-2022.03-1/bin/aarch64-linux-gcc)
set(CMAKE_CXX_COMPILER ${TOOL_CHAINS}/aarch64--glibc--stable-2022.03-1/bin/aarch64-linux-g++)

set(CMAKE_CUDA_STANDARD 11)

# 配置cuda的编译工具链nvcc
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")


# 设置编译选项 源代码反汇编视图用于在源代码和汇编指令级别显示内核的分析结果。为了能够查看内核源代码，您需要使用-lineinfo选项编译代码。如果不使用此编译器选项，则只显示反汇编视图。
set(CMAKE_CUDA_FLAGS "-lineinfo")

# cuda的目标架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

include_directories(/home/zyd/drive/drive-linux/filesystem/targetfs/usr/include)


# CMAKE_CUDA_HOST_COMPILER会选择编译CUDA语言文件的主机代码时要使用的编译器可执行文件。这映射到nvcc-ccbin选项
# set(CMAKE_CUDA_HOST_COMPILER "${ORIN_HOST}/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++")
set(CMAKE_CUDA_HOST_COMPILER "${TOOL_CHAINS}/aarch64--glibc--stable-2022.03-1/bin/aarch64-linux-g++")

```



x86_toolchain.cmake如下：

```cmake

set(CMAKE_CUDA_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

set(CMAKE_CUDA_FLAGS "-g -G -gencode arch=compute_61,code=sm_61 -lineinfo --default-stream=per-thread")
```





# 关于toolchain file

下面是一个交叉编译的 toolchain file

```cmake
 set(CMAKE_SYSTEM_NAME Linux)
 set(CMAKE_SYSTEM_PROCESSOR aarch64)
  
 set(target_arch aarch64-linux-gnu)
 set(CMAKE_LIBRARY_ARCHITECTURE ${target_arch} CACHE STRING "" FORCE)
  
 # Configure cmake to look for libraries, include directories and
 # packages inside the target root prefix.
 set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
 set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
 set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
 set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
 set(CMAKE_FIND_ROOT_PATH "/usr/${target_arch}")
  
 # needed to avoid doing some more strict compiler checks that
 # are failing when cross-compiling
 set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  
 # specify the toolchain programs
 find_program(CMAKE_C_COMPILER ${target_arch}-gcc)
 find_program(CMAKE_CXX_COMPILER ${target_arch}-g++)
 if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
     message(FATAL_ERROR "Can't find suitable C/C++ cross compiler for ${target_arch}")
 endif()
  
 set(CMAKE_AR ${target_arch}-ar CACHE FILEPATH "" FORCE)
 set(CMAKE_RANLIB ${target_arch}-ranlib)
 set(CMAKE_LINKER ${target_arch}-ld)
  
 # Not all shared libraries dependencies are instaled in host machine.
 # Make sure linker doesn't complain.
 set(CMAKE_EXE_LINKER_FLAGS_INIT -Wl,--allow-shlib-undefined)
  
 # instruct nvcc to use our cross-compiler
 set(CMAKE_CUDA_FLAGS "-ccbin ${CMAKE_CXX_COMPILER} -Xcompiler -fPIC" CACHE STRING "" FORCE)
```

这样使用

```bash
~/src$ cd build
~/src/build$ cmake -DCMAKE_TOOLCHAIN_FILE=~/Toolchain-eldk-ppc74xx.cmake ..
...
```

这个toolchain file有些是必须的有些不是必须的

必须的设置如下：

* `CMAKE_SYSTEM_NAME`：设置目标系统类型（如 Linux, WindowsCE, Android 等）

* `CMAKE_SYSTEM_PROCESSOR`：设置目标处理器架构（如 aarch64, arm, x86_64）

  * 如何知道应该写为什么呢？可以在目标架构上执行`uname -m` 输出就类似 `aarch64`

  * | 目标架构                 | 推荐设置 `CMAKE_SYSTEM_PROCESSOR` 为 |
    | ------------------------ | ------------------------------------ |
    | x86 (32-bit)             | `i686`, `i386`, `x86`                |
    | x86_64 (64-bit)          | `x86_64`                             |
    | ARM 32-bit (e.g., ARMv7) | `arm`, `armv7l`                      |
    | ARM 64-bit (AArch64)     | `aarch64`, `arm64`                   |
    | MIPS                     | `mips`, `mips64`                     |
    | PowerPC                  | `powerpc`, `ppc64`                   |

* CMAKE_C_COMPILER:  顾名思义，即C语言编译器，这里可以将变量设置成完整路径或者文件名，设置成完整路径有一个好处就是CMake会去这个路径下去寻找编译相关的其他工具比如linker,binutils等，如果你写的文件名带有arm-elf等等前缀，CMake会识别到并且去寻找相关的交叉编译器。
* CMAKE_CXX_COMPILER: 同上，此时代表的是C++编译器。

非必须的如下：

* CMAKE_FIND_ROOT_PATH:  代表了一系列的相关文件夹路径的根路径的变更，比如你设置了/opt/arm/,所有的Find_xxx.cmake都会优先根据这个路径下的/usr/lib,/lib等进行查找，然后才会去你自己的/usr/lib和/lib进行查找，如果你有一些库是不被包含在/opt/arm里面的，你也可以显示指定多个值给CMAKE_FIND_ROOT_PATH,比如

```
set(CMAKE_FIND_ROOT_PATH /opt/arm /opt/inst)
```

* CMAKE_FIND_ROOT_PATH_MODE_PROGRAM:  对FIND_PROGRAM()起作用，有三种取值，NEVER,ONLY,BOTH,第一个表示不在你CMAKE_FIND_ROOT_PATH下进行查找，第二个表示只在这个路径下查找，第三个表示先查找这个路径，再查找全局路径，对于这个变量来说，一般都是调用宿主机的程序，所以一般都设置成NEVER
* CMAKE_FIND_ROOT_PATH_MODE_LIBRARY: 对FIND_LIBRARY()起作用，表示在链接的时候的库的相关选项，因此这里需要设置成ONLY来保证我们的库是在交叉环境中找的.
* CMAKE_FIND_ROOT_PATH_MODE_INCLUDE:  对FIND_PATH()和FIND_FILE()起作用，一般来说也是ONLY,如果你想改变，一般也是在相关的FIND命令中增加option来改变局部设置，有NO_CMAKE_FIND_ROOT_PATH,ONLY_CMAKE_FIND_ROOT_PATH,BOTH_CMAKE_FIND_ROOT_PATH

## CMAKE_SYSROOT与CMAKE_FIND_ROOT_PATH

```cmake
# 设置 sysroot 路径
set(CMAKE_SYSROOT "/opt/toolchain/aarch64-linux-gnu/sysroot")

# 设置查找根路径（这里包含了 sysroot 和其他可能的库路径）
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT} /home/user/mylibs)

# 控制查找行为
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # 不在 root path 中查找可执行程序
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)    # 库文件只在 root path 中查找
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)    # 头文件只在 root path 中查找
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)    # 包配置文件只在 root path 中查找
```

- **`CMAKE_SYSROOT`**：主要用于编译器和链接器，告诉它们目标平台的标准库和头文件在哪里。 只能在toolchain file中设置
- **`CMAKE_FIND_ROOT_PATH`**：是一个更广泛的查找路径集合，不仅包含 `CMAKE_SYSROOT`，还可以包含其他你需要查找的路径。
- **配合使用**：通过设置 `CMAKE_FIND_ROOT_PATH_MODE_*` 变量，可以精确控制 CMake 如何在这些路径中查找不同的资源。

但是问题来了，有的时候我并没有设置CMAKE_SYSROOT，那么交叉编译联接的头文件和库文件是怎么知道使用的是arm版本呢？

例如下面的toolchain file .

```cmake
# 要为其生成CMake的操作系统的名称
set(CMAKE_SYSTEM_NAME Linux)
# 目标体系结构的CMake标识符。
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(TOOL_CHAINS "/opt/nvidia/driveos/toolchains")
# set(CMAKE_SYSROOT /home/zyd/drive/drive-linux/filesystem/targetfs)

# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
# set(CMAKE_STAGING_PREFIX /home/devel/stage)

# include_directories(/media/zyd/VisionPerception40Env/toolChainsQnx710/target/qnx7/usr/include)
include_directories(/home/zyd/drive/drive-linux/filesystem/targetfs/usr/include)

# 配置交叉编译工具链gcc和g++

set(CMAKE_C_COMPILER ${TOOL_CHAINS}/aarch64--glibc--stable-2022.03-1/bin/aarch64-linux-gcc)
set(CMAKE_CXX_COMPILER ${TOOL_CHAINS}/aarch64--glibc--stable-2022.03-1/bin/aarch64-linux-g++)

set(CMAKE_CUDA_STANDARD 11)

# 配置cuda的编译工具链nvcc
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")


# 设置编译选项 源代码反汇编视图用于在源代码和汇编指令级别显示内核的分析结果。为了能够查看内核源代码，您需要使用-lineinfo选项编译代码。如果不使用此编译器选项，则只显示反汇编视图。
set(CMAKE_CUDA_FLAGS "-lineinfo")

# cuda的目标架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

include_directories(/home/zyd/drive/drive-linux/filesystem/targetfs/usr/include)


# CMAKE_CUDA_HOST_COMPILER会选择编译CUDA语言文件的主机代码时要使用的编译器可执行文件。这映射到nvcc-ccbin选项
# set(CMAKE_CUDA_HOST_COMPILER "${ORIN_HOST}/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++")
set(CMAKE_CUDA_HOST_COMPILER "${TOOL_CHAINS}/aarch64--glibc--stable-2022.03-1/bin/aarch64-linux-g++")
```

只要你的交叉编译器正确配置了 `-sysroot` 或默认指向目标平台的 sysroot，那么像 `<stdlib.h>` 这样的系统头文件会从交叉编译器所使用的 **目标平台的 sysroot 中加载**，而不是主机上的。

当你使用交叉编译器（如 `aarch64-linux-gnu-gcc`）时，这个编译器本身已经配置好了对目标平台的支持，包括：

- 默认的系统头文件路径（如 `/usr/aarch64-linux-gnu/include/`）；
- 默认的库路径（如 `/usr/aarch64-linux-gnu/lib/`）；

这些路径是硬编码在交叉编译器中的，或者通过 `-sysroot` 自动传递的。



* 编译器本身就已经配置好了默认的系统头文件路径和库路径。你可以通过以下命令查看它默认搜索哪些路径：

  ```bash
  aarch64-buildroot-linux-gnu-gcc -E -v -x c++ -
  ```

* 如何确认当前编译器是否会使用正确的 sysroot

  ```bash
  aarch64-buildroot-linux-gnu-gcc -print-sysroot
  ```

* 看编译器使用的库路径，你可以运行

  ```bash
  aarch64-buildroot-linux-gnu-gcc -print-search-dirs
  ```

例如我的交叉编译gcc输出如下

可以看出，编译器自己已经设置了--with-sysroot 还有头文件和库文件的路径

```bash
root@P7479785A244:/home/zyd# /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/aarch64-buildroot-linux-gnu-gcc -E -v -x c++ -
Using built-in specs.
COLLECT_GCC=/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/aarch64-buildroot-linux-gnu-gcc.br_real
Target: aarch64-buildroot-linux-gnu
Configured with: ./configure --prefix=/sources/build/build --sysconfdir=/sources/build/build/etc --enable-static --target=aarch64-buildroot-linux-gnu --with-sysroot=/sources/build/build/aarch64-buildroot-linux-gnu/sysroot --enable-__cxa_atexit --with-gnu-ld --disable-libssp --disable-multilib --disable-decimal-float --with-gmp=/sources/build/build --with-mpc=/sources/build/build --with-mpfr=/sources/build/build --with-pkgversion='Buildroot 2020.08' --with-bugurl=http://bugs.buildroot.net/ --without-zstd --disable-libquadmath --disable-libquadmath-support --enable-tls --enable-threads --without-isl --without-cloog --with-abi=lp64 --with-cpu=cortex-a53 --enable-languages=c,c++,fortran --with-build-time-tools=/sources/build/build/aarch64-buildroot-linux-gnu/bin --enable-shared --enable-libgomp
Thread model: posix
gcc version 9.3.0 (Buildroot 2020.08) 
COLLECT_GCC_OPTIONS='-E' '-v' '-mcpu=cortex-a53' '-mlittle-endian' '-mabi=lp64'
 /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../libexec/gcc/aarch64-buildroot-linux-gnu/9.3.0/cc1plus -E -quiet -v -iprefix /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/ -isysroot /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot -D_GNU_SOURCE - -mcpu=cortex-a53 -mlittle-endian -mabi=lp64
ignoring duplicate directory "/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/../../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/include/c++/9.3.0"
ignoring duplicate directory "/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/../../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/include/c++/9.3.0/aarch64-buildroot-linux-gnu"
ignoring duplicate directory "/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/../../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/include/c++/9.3.0/backward"
ignoring duplicate directory "/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/../../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include"
ignoring nonexistent directory "/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot/usr/local/include"
ignoring duplicate directory "/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/../../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include-fixed"
ignoring duplicate directory "/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/../../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/include"
#include "..." search starts here:
#include <...> search starts here:
 /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/include/c++/9.3.0
 /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/include/c++/9.3.0/aarch64-buildroot-linux-gnu
 /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/include/c++/9.3.0/backward
 /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include
 /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/include-fixed
 /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/include
 /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot/usr/include
End of search list.
^C
root@P7479785A244:/home/zyd# /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/aarch64-buildroot-linux-gnu-gcc -print-sysroot
/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot
root@P7479785A244:/home/zyd# /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/aarch64-buildroot-linux-gnu-gcc -print-search-dirs
install: /opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/
programs: =/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../libexec/gcc/aarch64-buildroot-linux-gnu/9.3.0/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../libexec/gcc/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/bin/aarch64-buildroot-linux-gnu/9.3.0/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/bin/
libraries: =/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/lib/aarch64-buildroot-linux-gnu/9.3.0/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/lib/../lib64/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot/lib/aarch64-buildroot-linux-gnu/9.3.0/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot/lib/../lib64/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot/usr/lib/aarch64-buildroot-linux-gnu/9.3.0/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot/usr/lib/../lib64/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/bin/../lib/gcc/aarch64-buildroot-linux-gnu/9.3.0/../../../../aarch64-buildroot-linux-gnu/lib/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot/lib/:/opt/nvidia/driveos/toolchains/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot/usr/lib/
```



# 附录：

* Cmake官方：https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html
* toolchain file的例子：https://docs.nvidia.com/vpi/sample_cross_aarch64.html
* 博客 ：[CMake交叉编译配置](https://www.cnblogs.com/rickyk/p/3875334.html)