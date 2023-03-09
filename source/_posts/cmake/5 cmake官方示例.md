# 5 cmake官方示例

CMake教程通过一个示例项目, 逐步介绍了CMake构建系统的主要功能, 包括:

1. 生成执行程序
2. 生成库(使用旧式CMake宏定义方法)
3. 生成库(使用新式CMake宏定义方法--使用要求)
4. 基于源代码的安装与测试
5. 系统检测
6. 添加自定义命令以及生成文件
7. 构建安装程序
8. 将测试结果提交到Kitware的公共指示板
9. 混合静态和共享(动态)库
10. 生成器表达式,条件判断
11. [find_package()](https://cmake.org/cmake/help/latest/command/find_package.html#command:find_package)支持(添加导出配置)
12. 将调试版和发行版打包在一起(win10试验没有成功)

> 注:
>
> 1. 示例项目目录在CMake源代码树的`Help\guide\tutorial`目录
> 2. 本摘要主要关注CMake使用, 一些关于源代码的更改在请查看[CMake教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
> 3. CMake教程示例代码,下一步的内容都是上一步的结果

## 主要功能摘要

## 5.1. 生成执行程序

本节示例展示了:

- 生成执行程序的基本构建系统框架;
- 将一些CMake属性通过配置文件输出;
- 如何在目标中添加头文件搜索路径;
- 常规CMake命令行使用方式

> 注: 本节示例初始代码在示例目录的`Step1`子目录, 成果在`Step2`子目录

#### 核心代码

顶层目录的**CMakeLists.txt**内容如下:

```cmake
# 指定CMake最小版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称及版本号
project(Tutorial VERSION 1.0)

# 指定C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 配置头文件, 将一些CMake属性值保存到源代码中
configure_file(TutorialConfig.h.in TutorialConfig.h)

# 生成执行程序
add_executable(Tutorial tutorial.cxx)

# 将项目输出目录(binary tree)添加到Tutorial目标的头文件搜索路径, 这样才能找到生成的TutorialConfig.h
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```

**TutorialConfig.h.in**内容如下:

```c
// 接受CMake配置属性
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
```

#### 命令行使用示例

在项目目录下

```cli
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```

## 5.2. 生成库(旧式CMake)

在上一步的基础上, 本节示例展示了:

- 生成库;
- 在项目中添加库;
- 使用可选项,并将可选项保存到配置文件中;
- 目标链接库的旧式方法;
- 目标包含库头文件目录的旧式方法;

> 注: 本节示例初始代码在示例目录的`Step2`子目录, 成果在`Step3`子目录

#### 生成库

1. 创建库目录;
2. 在库目录**CMakeLists.txt**中, 添加生成库命令如下:

```cmake
# 生成库
add_library(MathFunctions mysqrt.cxx)
```

#### 其它核心代码

项目**CMakeLists.txt**内容如下:

```cmake
# 指定CMake最小版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称及版本号
project(Tutorial VERSION 1.0)

# 指定C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 是否使用自定义数学库可选项
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# 配置头文件, 将一些CMake属性值保存到源代码中
configure_file(TutorialConfig.h.in TutorialConfig.h)

# 添加自定义数学库
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

# 生成执行程序
add_executable(Tutorial tutorial.cxx)

# Tutorial目标链接相关库
target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

# Tutorial目标添加头文件搜索路径(项目生成目录,其它头文件目录)
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ${EXTRA_INCLUDES}
                           )
```

**TutorialConfig.h.in**内容如下:

```c
// 接受CMake配置属性
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
#cmakedefine USE_MYMATH
```

## 5.3. 生成库(新式CMake)

在上一步的基础上, 本节示例展示了:

- 在库中使用**INTERFACE**;
- 在项目中添加库;
- 使用可选项,并将可选项保存到配置文件中;
- 目标链接库的旧式方法;
- 目标包含库头文件目录的旧式方法;

> 注: 本节示例初始代码在示例目录的`Step3`子目录, 成果在`Step4`子目录

#### 核心代码更改

**库目录CMakeLists.txt**内容如下:

```cmake
# 生成库
add_library(MathFunctions mysqrt.cxx)

# 设置目录头文件搜索路径. INTERFACE是使用者需要, 库本身不需要
# 通过INTERFACE可以将头文件搜索路径转递给使用者
target_include_directories(MathFunctions
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
```

**项目CMakeLists.txt**删除了*EXTRA_INCLUDES*相关代码, 变化如下:

```cmake
# 指定CMake最小版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称及版本号
project(Tutorial VERSION 1.0)

# 指定C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 是否使用自定义数学库可选项
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# 配置头文件, 将一些CMake属性值保存到源代码中
configure_file(TutorialConfig.h.in TutorialConfig.h)

# 添加自定义数学库
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
endif()

# 生成执行程序
add_executable(Tutorial tutorial.cxx)

# Tutorial目标链接相关库(此代码会使用新式CMake方法传递库属性)
target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

# Tutorial目标添加头文件搜索路径
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```

## 5.4. 基于源代码的安装与测试

在上一步的基础上, 本节示例展示了:

- 使用基于源代码的安装规则(install);
- 使用测试;

> 注: 本节示例初始代码在示例目录的`Step4`子目录, 成果在`Step5`子目录

#### 核心代码更改

**库目录CMakeLists.txt**变化如下:

```cmake
# 生成库
add_library(MathFunctions mysqrt.cxx)

# 设置目录头文件搜索路径. INTERFACE是使用者需要, 库本身不需要
# 通过INTERFACE可以将头文件搜索路径转递给使用者
target_include_directories(MathFunctions
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

# 安装规则
install(TARGETS MathFunctions DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
```

**项目CMakeLists.txt**新增了安装与测试相关代码, 变化如下:

```cmake
# 指定CMake最小版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称及版本号
project(Tutorial VERSION 1.0)

# 指定C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 是否使用自定义数学库可选项
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# 配置头文件, 将一些CMake属性值保存到源代码中
configure_file(TutorialConfig.h.in TutorialConfig.h)

# 添加自定义数学库
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
endif()

# 生成执行程序
add_executable(Tutorial tutorial.cxx)

# Tutorial目标链接相关库(此代码会使用新式CMake方法传递库属性)
target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

# Tutorial目标添加头文件搜索路径
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

# 添加安装目标
install(TARGETS Tutorial DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  DESTINATION include
  )

# 开启测试
enable_testing()

# 验证Tutorial正常运行
add_test(NAME Runs COMMAND Tutorial 25)

# 验证Tutorial的使用方法提示
add_test(NAME Usage COMMAND Tutorial)
set_tests_properties(Usage
  PROPERTIES PASS_REGULAR_EXPRESSION "Usage:.*number"
  )

# 定义函数以简化测试添加
function(do_test target arg result)
  add_test(NAME Comp${arg} COMMAND ${target} ${arg})
  set_tests_properties(Comp${arg}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result}
    )
endfunction(do_test)

# 批量添加基于结果的测试
do_test(Tutorial 4 "4 is 2")
do_test(Tutorial 9 "9 is 3")
do_test(Tutorial 5 "5 is 2.236")
do_test(Tutorial 7 "7 is 2.645")
do_test(Tutorial 25 "25 is 5")
do_test(Tutorial -25 "-25 is [-nan|nan|0]")
do_test(Tutorial 0.0001 "0.0001 is 0.01")
```

#### 命令行使用示例

将调试版**安装**到*D:\Temp2\ttt\debug*

```cli
$ mkdir build
$ cd build
$ cmake ..
生成并安装调试版
$ cmake --build .
$ cmake --install . --prefix D:\Temp2\ttt\debug --config Debug
生成并安装发行版
$ cmake --build . -- config Release
$ cmake --install . --prefix D:\Temp2\ttt\Release --config Release
```

**win10 VS2019 测试**

在**build**目录测试应用程序

```cli
$ mkdir build
$ cd build
$ cmake ..
生成并测试调试版
$ cmake --build .
$ ctest -C Debug [-vv]
生成并测试发行版
$ cmake --build . -- config Release
$ ctest -C Release [-vv]
```

## 5.5. 系统检测

在上一步的基础上, 本节示例展示了:

- CMake系统功能检测, 以判断平台是否支持特定功能;
- 目标编译定义(宏定义);

> 注1: 本节示例初始代码在示例目录的`Step5`子目录, 成果在`Step6`子目录
> 注2: 在win10下测试失败, 没有发现log,exp函数

#### 核心代码更改

**库目录CMakeLists.txt**变化如下:

```cmake
# 生成库
add_library(MathFunctions mysqrt.cxx)

# 设置目录头文件搜索路径. INTERFACE是使用者需要, 库本身不需要
# 通过INTERFACE可以将头文件搜索路径转递给使用者
target_include_directories(MathFunctions
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

# 检测系统是否支持log,exp函数?
include(CheckSymbolExists)
set(CMAKE_REQUIRED_LIBRARIES "m")
check_symbol_exists(log "math.h" HAVE_LOG)
check_symbol_exists(exp "math.h" HAVE_EXP)

if(HAVE_LOG AND HAVE_EXP)
  # 目标宏定义
  target_compile_definitions(MathFunctions
                             PRIVATE "HAVE_LOG" "HAVE_EXP")
endif()

# 安装规则
install(TARGETS MathFunctions DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
```

## 5.6. 添加自定义命令以及生成文件

在上一步的基础上, 本节示例展示了:

- CMake使用命令;
- 使用生成文件;

> 注: 本节示例初始代码在示例目录的`Step6`子目录, 成果在`Step7`子目录

#### 核心代码更改

**库目录CMakeLists.txt**变化如下:

```cmake
# 首先生成一个用于产生预定义值表源代码的执行程序
add_executable(MakeTable MakeTable.cxx)

# 添加自定义命令以产生源代码(依赖于MakeTable目标)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
  COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
  DEPENDS MakeTable
  )

# 生成主库(包含上一步中生成的预定义值表源代码)
add_library(MathFunctions
            mysqrt.cxx
            ${CMAKE_CURRENT_BINARY_DIR}/Table.h
            )

# 通过INTERFACE向使用者传播头文件搜索路径
# 通过PRIVATE定义目标内部使用头文件搜索路径
target_include_directories(MathFunctions
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
          )

# 安装规则
install(TARGETS MathFunctions DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
```

## 5.7. 构建安装程序

在上一步的基础上, 本节示例展示了:

- 制作安装包;

> 注1: 本节示例初始代码在示例目录的`Step7`子目录, 成果在`Step8`子目录
> 注2: 为测试需要, 在win10上, 首先安装NSIS(Nulsoft Install System).

#### 核心代码更改

**顶层目录CMakeLists.txt**末尾添加如下内容即可:

```cmake
# 设置安装器
include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${Tutorial_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Tutorial_VERSION_MINOR}")
include(CPack)
```

#### 命令行使用示例

在项目目录下

```cli
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
VS generator 注意：目录不能含有中文
生成发行版安装包(Tutorial-1.0-win64.exe)
$ cmake --build . --target PACKAGE --config Release
生成调试版安装包(Tutorial-1.0-win64.exe)
$ cmake --build . --target PACKAGE --config Debug
```

## 5.8. 将测试结果提交到Kitware的公共指示板

在上一步的基础上, 本节示例展示了:

- 将测试结果提交到Kitware的公共指示板;

> 注: 本节示例初始代码在示例目录的`Step8`子目录, 成果在`Step9`子目录

#### 核心代码更改

**顶层目录CMakeLists.txt**将:

```cmake
# enable testing
enable_testing()
```

替换为:

```cmake
# enable dashboard scripting
include(CTest)
```

在顶层目录创建**CTestConfig.cmake**:

```cmake
set(CTEST_PROJECT_NAME "CMakeTutorial")
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=CMakeTutorial")
set(CTEST_DROP_SITE_CDASH TRUE)
```

#### 命令行使用示例

在项目目录下

```cli
$ mkdir build
$ cd build
$ cmake ..
VS generator 注意：目录不能含有中文
发布发行版测试结果
$ ctest [-VV] -C Release -D Experimental
发布调试版测试结果
$ ctest [-VV] -C Debug -D Experimental
```

## 5.9. 混合静态和共享库

在上一步的基础上, 本节示例展示了:

- 在动态库中使用静态库
- 使用`BUILD_SHARED_LIBS`属性设置库默认生成类型(共享/非共享)(默认生成静态库);
- 使用目标宏定义(target_compile_definitions)设置windows动态库标志(`__declspec(dllexport),__declspec(dllimport)`)
- 在add_library中控制库生成类型[STATIC | SHARED | MODULE]

> 注: 本节示例初始代码在示例目录的`Step9`子目录, 成果在`Step10`子目录

#### 核心代码更改

**顶层CMakeLists.txt**的起始部分更改为:

```cmake
# 指定CMake最小版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称及版本号
project(Tutorial VERSION 1.0)

# 指定C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 控制动静态库生成目录,这样在windows上不用考虑运行时路径问题
# we don't need to tinker with the path to run the executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# 配置头文件, 将一些CMake属性值保存到源代码中
configure_file(TutorialConfig.h.in TutorialConfig.h)

# 添加自定义数学库
add_subdirectory(MathFunctions)

# 生成执行程序
add_executable(Tutorial tutorial.cxx)
target_link_libraries(Tutorial PUBLIC MathFunctions)
```

**MathFunctions/CMakeLists.txt**更改如下:

```cmake
# 生成运行时需要的动态库
add_library(MathFunctions MathFunctions.cxx)

# 通过INTERFACE向使用者传播头文件搜索路径
target_include_directories(MathFunctions
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           )

# 是否使用自定义数学库
option(USE_MYMATH "Use tutorial provided math implementation" ON)
if(USE_MYMATH)
  # 数学库目标宏定义
  target_compile_definitions(MathFunctions PRIVATE "USE_MYMATH")

  # 首先生成一个用于产生预定义值表源代码的执行程序
  add_executable(MakeTable MakeTable.cxx)

  # 添加自定义命令以产生源代码(依赖于MakeTable目标)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    DEPENDS MakeTable
    )

  # 生成仅进行sqrt计算的静态库
  add_library(SqrtLibrary STATIC
              mysqrt.cxx
              ${CMAKE_CURRENT_BINARY_DIR}/Table.h
              )

  # 通过PRIVATE定义目标内部使用头文件搜索路径
  # Table.h文件生成在${CMAKE_CURRENT_BINARY_DIR}中
  target_include_directories(SqrtLibrary PRIVATE
                             ${CMAKE_CURRENT_BINARY_DIR}
                             )

  # 设置SqrtLibrary目标属性: 如果默认生成共享库, 需要将PIC(POSITION_INDEPENDENT_CODE)设置为True
  set_target_properties(SqrtLibrary PROPERTIES
                        POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
                        )

  # 将数学库链接到自定义sqrt库
  target_link_libraries(MathFunctions PRIVATE SqrtLibrary)
endif()

# 定义目标宏用于在windows上设置动态库标志declspec(dllexport)
target_compile_definitions(MathFunctions PRIVATE "EXPORTING_MYMATH")

# 安装规则
install(TARGETS MathFunctions DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
```

**MathFunctions/MathFunctions.h**使用dll导出(dll export)定义:

```cpp
#if defined(_WIN32)
#  if defined(EXPORTING_MYMATH)
#    define DECLSPEC __declspec(dllexport)
#  else
#    define DECLSPEC __declspec(dllimport)
#  endif
#else // non windows
#  define DECLSPEC
#endif

namespace mathfunctions {
double DECLSPEC sqrt(double x);
}
```

其它对C++代码的修改见[CMake教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)

#### 命令行使用示例

在项目目录下

```cli
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .

  Checking Build System
  Building Custom Rule E:/Help/Other/CMake/CMake_Tutorial/guide/tutorial/Step10/MathFunctions/CMakeLists.txt
  MakeTable.cxx
  MakeTable.vcxproj -> E:\Help\Other\CMake\CMake_Tutorial\guide\tutorial\Step10\build\Debug\MakeTable.exe
  Generating Table.h
  Building Custom Rule E:/Help/Other/CMake/CMake_Tutorial/guide/tutorial/Step10/MathFunctions/CMakeLists.txt
  mysqrt.cxx
  SqrtLibrary.vcxproj -> E:\Help\Other\CMake\CMake_Tutorial\guide\tutorial\Step10\build\Debug\SqrtLibrary.lib
  Building Custom Rule E:/Help/Other/CMake/CMake_Tutorial/guide/tutorial/Step10/MathFunctions/CMakeLists.txt
  MathFunctions.cxx
    正在创建库 E:/Help/Other/CMake/CMake_Tutorial/guide/tutorial/Step10/build/Debug/MathFunctions.lib 和对象 E:/Help/Other/CMake/CMake_Tutorial/guide/tutorial/St
  ep10/build/Debug/MathFunctions.exp
  MathFunctions.vcxproj -> E:\Help\Other\CMake\CMake_Tutorial\guide\tutorial\Step10\build\Debug\MathFunctions.dll
  Building Custom Rule E:/Help/Other/CMake/CMake_Tutorial/guide/tutorial/Step10/CMakeLists.txt
  tutorial.cxx
  Tutorial.vcxproj -> E:\Help\Other\CMake\CMake_Tutorial\guide\tutorial\Step10\build\Debug\Tutorial.exe
  Building Custom Rule E:/Help/Other/CMake/CMake_Tutorial/guide/tutorial/Step10/CMakeLists.txt
```

## 5.10. 生成器表达式,条件判断

在上一步的基础上, 本节示例展示了:

- 在构建过程中计算生成器表达式值，以生成特定于每个构建配置的信息
- 生成器表达式有三类: 逻辑表达式、信息表达式和输出表达式(Logical, Informational, and Output expressions)
- 逻辑表达式: `$<0:...>`的结果是空字符串，`$<1:...>`的结果是“...”的内容。它们也可以嵌套。
- 生成器表达式通常用来按需添加编译标志
- 使用**INTERFACE目标**在目标间传递属性

> 注: 本节示例初始代码在示例目录的`Step10`子目录, 成果在`Step11`子目录

#### 核心代码更改

**顶层CMakeLists.txt**的相应部分:

```cmake
# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```

更改为:

```cmake
add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_11)

# 通过BUILD_INTERFACE生成器表达式添加编译器警告标志
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
target_compile_options(tutorial_compiler_flags INTERFACE
  "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
  "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
)
```

**库目录CMakeLists.txt**目标链接库代码:

```cmake
target_link_libraries(MakeTable[SqrtLibrary|MathFunctions] PRIVATE[PUBLIC] tutorial_compiler_flags)
```

## 5.11. 添加导出配置,以支持find_package

在上一步的基础上, 本节示例展示了:

- 如何让其他项目方便地使用本项目
- 在**install(TARGETS)**中使用**EXPORT**关键字, 它用于将目标从安装树导入到另一个项目
- 生成**MathFunctionsConfig.cmake**, 以让[find_package()](https://cmake.org/cmake/help/latest/command/find_package.html#command:find_package)命令可以找到本项目
- 显式安装生成的**MathFunctionsTargets.cmake**

> 注: 本节示例初始代码在示例目录的`Step11`子目录, 成果在`Step12`子目录

#### 核心代码更改

在**MathFunctions/CMakeLists.txt**中增加**EXPORT**:

```cmake
# 根据项目内与安装方式包含不同的头文件目录
target_include_directories(MathFunctions
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )

# 此处省略其他代码...

# 安装规则
install(TARGETS MathFunctions tutorial_compiler_flags
        DESTINATION lib
        EXPORT MathFunctionsTargets)
install(FILES MathFunctions.h DESTINATION include)
```

在顶层目录中添加**Config.cmake.in**, 用来生成**MathFunctionsTargets.cmake**:

```cmake
@PACKAGE_INIT@

include ( "${CMAKE_CURRENT_LIST_DIR}/MathFunctionsTargets.cmake" )
```

在**顶层CMakeLists.txt**，为了正确配置和安装**MathFunctionsTargets.cmake**, 添加以下内容:

```cmake
install(EXPORT MathFunctionsTargets
  FILE MathFunctionsTargets.cmake
  DESTINATION lib/cmake/MathFunctions
)

include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/example"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )
# generate the version file for the config file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfigVersion.cmake"
  VERSION "${Tutorial_VERSION_MAJOR}.${Tutorial_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsConfig.cmake
  DESTINATION lib/cmake/MathFunctions
  )
```

到这里，这是一个可以在项目安装或打包以后重定位的CMake配置。 如果需要项目在一个构建目录中使用，在顶层**CMakeLists.txt**的结尾添加以下内容:

```cmake
export(EXPORT MathFunctionsTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsTargets.cmake"
)
```

> 通过这个导出调用，我们现在生成一个`Targets.cmake`，使得在构建目录中配置生成的`MathFunctionsConfig.cmake`可以被其他项目使用，而不需要安装它。
> 此处未试验, 不理解

## 5.12. 将调试版和发行版打包在一起

在上一步的基础上, 本节示例展示了:

- 为调试版文件设置后缀: 非执行程序目标会根据**CMAKE_DEBUG_POSTFIX**生成它的**_POSTFIX**
- 执行程序目录的**DEBUG_POSTFIX**需要手工设置

> 注1: 本节示例初始代码在示例目录的`Step12`子目录, 成果在`Complete`子目录
> 注2: 本节打包工作在win10上没有成功

### 核心代码更改

在**顶层CMakeLists.txt**中更改:

```cmake
# 设置非可执行程序目标的调试后缀
set(CMAKE_DEBUG_POSTFIX d)
...
# 设置可执行程序目标的调试后缀
set_target_properties(Tutorial PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
```





