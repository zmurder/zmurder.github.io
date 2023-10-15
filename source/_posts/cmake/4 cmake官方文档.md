# 4 cmake官方文档

CMake 教程提供了一个循序渐进的指南，涵盖了 CMake 帮助解决的常见构建系统问题。了解示例项目中各个主题是如何运作是很有帮助的。教程的文档和示例代码可以在 CMake 源代码的 `Help/guide/tutorial` 目录中找到。每个步骤都有自己的子目录，其中包含可以用作起点的代码。教程示例是渐进的，因此，每个步骤都为前一个步骤提供完整的解决方案。

## 4.1 一个基本的出发点（第一步）

最基本的项目是一个从源代码文件构建的可执行文件。对于简单的项目来说，仅需一份包含三行内容的 `CMakeLists.txt` 文件。这将是我们教程的起点。在 `Step1` 文件夹创建一个 `CMakeLists.txt` 文件，内容如下：

```cmake
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(Tutorial)

# 生成可执行文件 Tutorial
add_executable(Tutorial tutorial.cxx)
```

请注意：在这个实例中，我们使用小写字母在 `CMakeLists.txt` 中书写命令。CMake 支持大写、小写以及大小写混合的命令形式。`tutorial.cxx` 文件的源码在 Step1 目录中，它可以用于计算一个数字的平方根。

## 添加一个版本号和配置的头文件

我们要添加的第一个特性是为可执行文件和项目提供一个版本号。尽管我们可以在源代码中执行这一操作，但是使用 `CMakeLists.txt` 可以提供更大的灵活性。

首先，修改 `CMakeLists.txt` 文件，使用 `project()` 命令来设置项目的名称和版本号。

```cmake
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(Tutorial VERSION 1.0)
```

然后，配置头文件来将版本号传递给源代码。

```cmake
configure_file(TutorialConfig.h.in TutorialConfig.h)
```

由于配置好的文件将被写入二进制目录，所以我们必须将该目录添加到 include 文件的搜索路径中。将以下内容添加到 `CMakeLists.txt` 文件的末尾。

```cmake
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```

使用你喜欢的编辑器，在源代码目录创建 `TutorialConfig.h.in` 文件，内容如下：

```cpp
// Tutorial 的配置选项和设置
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
```

当 CMake 配置这个头文件的时候，`@Tutorial_VERSION_MAJOR@` 和 `@Tutorial_VERSION_MINOR@` 的值将被替换。

接下来，修改 `tutorial.cxx` 文件，让它 include 那个配置过的头文件：`TutorialConfig.h`。

最后，让我们通过修改 `tutorial.cxx` 来打印出可执行文件的名称和版本号，如下所示：

```cpp
if (argc < 2) {
    // 打印版本号
    std::cout << argv[0] << " Version " << Tutorial_VERSION_MAJOR << "."
              << Tutorial_VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }
```

## 指定 C++ 标准

接下来，让我们通过在 `tutorial.cxx` 中用 `std::stod` 替换 `atof`，来将一些 C++ 11 的特性添加到我们的项目中。同时，删除 `#include <cstdlib>`。

```cpp
const double inputValue = std::stod(argv[1]);
```

我们需要在 CMake 代码中显式地声明它需要使用正确的标识。在 CMake 中，启用对特定 C++ 标准支持的最简单做法是使用 `CMAKE_CXX_STANDARD` 变量。在本教程中，我们在 `CMakeLists.txt` 文件中将 `CMAKE_CXX_STANDARD` 变量设置为 11，并将 `CMAKE_CXX_STANDARD_REQUIRED` 变量设置为 True。确保将 `CMAKE_CXX_STANDARD` 声明添加到 `add_executable` 命令的上方。

```cmake
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(Tutorial VERSION 1.0)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```

## 构建并测试

运行 `cmake` 可执行文件或是 `cmake-gui` 来配置项目，然后使用你选择的构建工具进行构建。

例如，我们可以使用命令行导航到 CMake 源代码的 `Help/guide/tutorial` 目录，并创建一个构建目录。

```powershell
mkdir Step1_build
```

接下来，导航到构建目录并运行 CMake 来配置项目并生成一个本地构建系统。

```powershell
cd Step1_build
cmake ../Step1
```

然后调用构建系统来实际地编译和链接项目：

```powershell
#在 linux 下就是执行 make 命令。 
cmake --build .
```

最后，尝试通过下列命令使用新构建的 Tutorial：

```powershell
Tutorial 4294967296
Tutorial 10
Tutorial
```

## 4.2 添加一个库（第二步）

现在我们将向我们的项目中添加一个库。这个库将包含我们自己的计算数字平方根的实现。然后，可执行文件可以使用这个库，而非编译器提供的标准平方根函数。

**也就是先把自己实现的函数（文件）生成一个库，由另一个文件调用。**

在本教程中，我们将把库放到一个名为 `MathFunctions` 的子目录中。这个目录已经包含了一个头文件：`MathFunctions.h` 和一个源文件 `mysqrt.cxx`。源文件中有一个名为 `mysqrt` 的函数，它提供了和编译器的 `sqrt` 函数类似的功能。

将具有以下内容的 `CMakeLists.txt` 文件添加到 `MathFunctions` 目录中。

```cmake
# 添加一个子目录并构建该子目录
add_library(MathFunctions mysqrt.cxx)
```

为了使用这个新的库，我们将在顶级的 `CMakeLists.txt` 文件中添加一个 `add_subdirectory()` 的命令，以便构建该库。我们将新库添加到可执行文件，并将 MathFunctions 添加为 include 目录，以便可以找到 `mysqrt.h` 头文件。顶级 `CMakeLists.txt` 的最后几行现在应该是这样的：

```cmake
#添加一个子目录并构建该子目录
add_subdirectory(MathFunctions)

# 生成可执行文件
add_executable(Tutorial tutorial.cxx)
# 可执行程序 依赖MathFunctions库
target_link_libraries(Tutorial PUBLIC MathFunctions)

# 将二进制目录添加到 include 文件的搜索路径
# 这样我们就可以找到 TutorialConfig.h 了
target_include_directories(Tutorial PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MathFunctions"
                          )
```

现在让我们将 `MathFunctions` 库设为可选的。虽然对于本教程来说没有必要这么做，但是对于大型项目而言这是很常见的情况。第一步是向顶层的 `CMakeLists.txt` 文件添加一个选项。

```cmake
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# 配置一个头文件来将某些 CMake 设置
# 传递给源码
configure_file(TutorialConfig.h.in TutorialConfig.h)
```

这个选项将会在 `cmake-gui` 和 `ccmake` 中显示，默认值 `ON` 可以由用户更改。这个设置将被存储在缓存中，这样用户就无需每次在构建目录上运行 CMake 时都设置该值。

下一个修改是让构建和链接 MathFunctions 库成为有条件的。为此，我们将顶级 `CMakeLists.txt` 文件的结尾修改为如下的形式：

```cmake
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  #将MathFunctions追加到变量EXTRA_LIBS
  list(APPEND EXTRA_LIBS MathFunctions)
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

# 添加可执行文件
add_executable(Tutorial tutorial.cxx)

target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

# 配置一个头文件来将某些 CMake 设置
# 传递给源码
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ${EXTRA_INCLUDES}
                           )
```

请注意：使用变量 `EXTRA_LIBS` 来收集所有可选库，以便稍后连接到可执行文件中。变量 `EXTRA_INCLUDES` 类似地用于可选的头文件。这是处理许多可选组件的经典方法，我们将在下一步讨论现代方法。

对源码的相应更改非常简单。首先，如果我们需要，在 `tutorial.cxx` 中包含 `MathFunctions.h` 标头。

```cpp
#ifdef USE_MYMATH
#  include "MathFunctions.h"
#endif
```

然后，还是这个文件中，让 `USE_MYMATH` 控制使用哪个平方根函数。

```cpp
#ifdef USE_MYMATH
  const double outputValue = mysqrt(inputValue);
#else
  const double outputValue = sqrt(inputValue);
#endif
```

由于源码现在需要使用 `USE_MYMATH`，因此我们可以使用下列内容将其添加到 `TutorialConfig.h.in` 中：

```cpp
#cmakedefine USE_MYMATH
```

**练习**：为什么在 `USE_MYMATH` 选项之后配置 `TutorialConfig.h.in` 如此重要？如果我们将二者颠倒会发生什么？

下面是完整的CMakeLists.txt

```cmake
#顶层CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Tutorial VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# should we use our own math functions
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file(TutorialConfig.h.in TutorialConfig.h)

# add the MathFunctions library
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

# add the executable
add_executable(Tutorial tutorial.cxx)

target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ${EXTRA_INCLUDES}
                           )
```

```cmake
#MathFunctions子文件夹的CMakeLists.txt
add_library(MathFunctions mysqrt.cxx)
```

```c
//TutorialConfig.h.in
// the configured options and settings for Tutorial
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
#cmakedefine USE_MYMATH
```



运行 `cmake` 可执行文件或者 `cmake-gui` 来配置项目，然后使用你选择的构建工具对其进行构建。接着运行被构建好的 Tutorial 可执行文件。

现在，让我们更新 `USE_MYMATH` 的值。最简单的方法是在终端使用 `cmake-gui` 或者 `ccmake`。或者，如果你想从命令行对选项进行修改，尝试以下命令：

```powershell
cmake ../Step2 -DUSE_MYMATH=OFF
```

重新构建并再次运行。

哪个函数提供的结果更佳，sqrt 还是 mysqrt ？

## 4.3 为库添加使用要求（第三步）

使用要求可以更好地控制库或可执行文件的链接以及 include 内容，同时还可以更好地控制 CMake 内部目标的传递属性。使用使用要求的主要命令有：

- `target_compile_definitions()`

- `target_compile_options()`

- `target_include_directories()`

- `target_link_libraries()`

让我们从添加一个库（第二步）的基础上重构代码，以使用具有使用要求的现代 CMake 方法。我们首先声明，任何连接到 MathFunctions 的人都必须 include 当前的源目录，而 MathFunctions 本身不需要。因此，这可以成为一个 `INTERFACE` 使用要求。

请记住，`INTERFACE` 指的是使用者需要而提供者不需要的东西。将以下内容添加到 `MathFunctions/CMakeLists.txt` 的末尾：

```cmake
target_include_directories(MathFunctions
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )
```

现在，我们已经指定了 MathFunctions 的使用要求，我们可以安全地从顶级 `CMakeLists.txt` 中删除对 `EXTRA_INCLUDES` 变量的使用，这里：

```cmake
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
endif()
```

还有这里：

```cmake
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```

完整的CMakeList.txt如下

顶级 `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(Tutorial VERSION 1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# should we use our own math functions
option(USE_MYMATH "Use tutorial provided math implementation" ON)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file(TutorialConfig.h.in TutorialConfig.h)

# add the MathFunctions library
if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
  #删除了list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")

endif()

# add the executable
add_executable(Tutorial tutorial.cxx)

target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           #删除了${EXTRA_INCLUDES}
                           )
```

`MathFunctions/CMakeLists.txt` 

```cmake
add_library(MathFunctions mysqrt.cxx)

# state that anybody linking to us needs to include the current source dir
# to find MathFunctions.h, while we don't.
# 下面的语句是多出来的
target_include_directories(MathFunctions
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          )

```

完成之后，运行 `cmake` 可执行文件或者 `cmake-gui` 来配置项目，然后从构建目录使用你选择的构建工具或者使用 `cmake --build .` 进行构建。

## 4.4 安装并测试（第四步）

现在，我们可以开始向我们的项目中添加安装规则和测试支持了。

## 安装规则

安装规则十分简单：对于 MathFunctions，我们要安装库和头文件，对于应用程序，我们要安装可执行文件和配置的头文件。

因此，我们要在 MathFunctions/CMakeLists.txt 的末尾添加：

```cmake
install(TARGETS MathFunctions DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
```

在顶级 CMakeLists.txt 的末尾添加：

```cmake
install(TARGETS Tutorial DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  DESTINATION include
  )
```

这就是创建本教程的基本本地安装所需的全部内容。

现在，运行 cmake 可执行文件或者 cmake-gui 来配置项目，然后使用你选择的构建工具对其进行构建。

然后，通过命令行使用 `cmake` 命令中的 `install` 选项（这一选项在 3.15 中被引入，较早版本的 CMake 必须使用 `make install`）运行安装步骤。对于多配置工具，记得使用 `--config` 参数来指定配置。如果你使用 IDE，只需要构建 `INSTALL` 目标。这一步将安装适当的头文件、库和可执行文件。例如：

```powershell
cmake --install .
```

CMake 变量 `CMAKE_INSTALL_PREFIX` 被用于确定文件安装的根目录。如果使用 `cmake --install` 命令，则可以通过 `--prefix` 参数修改安装目录。例如：

```powershell
cmake --install . --prefix "/home/myuser/installdir"
```

导航到安装目录，并验证已安装的 Tutorial 是否运行。

## 测试支持

接下来测试我们的应用程序。在顶级 `CMakeLists.txt` 文件的末尾，我们可以启用测试，然后添加一些基本测试以验证应用程序是否正常运行。

```cmake
enable_testing()

# 应用程序是否运行
add_test(NAME Runs COMMAND Tutorial 25)

# 用法消息有效吗？
add_test(NAME Usage COMMAND Tutorial)
set_tests_properties(Usage
  PROPERTIES PASS_REGULAR_EXPRESSION "Usage:.*number"
  )

# 定义一个函数以简化添加测试
function(do_test target arg result)
  add_test(NAME Comp${arg} COMMAND ${target} ${arg})
  set_tests_properties(Comp${arg}
    PROPERTIES PASS_REGULAR_EXPRESSION ${result}
    )
endfunction(do_test)

# 做一些基于结果的测试
do_test(Tutorial 4 "4 is 2")
do_test(Tutorial 9 "9 is 3")
do_test(Tutorial 5 "5 is 2.236")
do_test(Tutorial 7 "7 is 2.645")
do_test(Tutorial 25 "25 is 5")
do_test(Tutorial -25 "-25 is [-nan|nan|0]")
do_test(Tutorial 0.0001 "0.0001 is 0.01")
```

第一个测试只是验证应用程序正在运行，没有段错误或其他崩溃，并且返回值为零。这是 CTest 测试的基本形式。

下一个测试使用 `PASS_REGULAR_EXPRESSION` 测试属性来验证测试的输出是否包含某些字符串。在这种情况下验证在提供了错误数量的参数时是否打印了使用情况消息。

最后，我们有一个名为 `do_test` 的函数，该函数运行应用程序并验证所计算的平方根相对于给定输入是否正确。对于 `do_test` 的每次调用，都会根据传递的参数将另一个测试（带有名称，输入和预期结果）添加到项目中。

重建应用程序，然后导航到二进制目录并运行 `ctest` 可执行文件：`ctest -N` 和 `ctest -VV`。对于多配置生成器（例如 Visual Studio），必须指定配置类型。例如，要以 Debug 模式运行测试，请从构建目录（而非 Debug 子目录）中使用 `ctest -C Debug -VV`。或者，从 IDE 构建 `RUN_TESTS` 目标。

## 4.5 添加系统自检（第五步）

让我们考虑向我们的项目中添加一些依赖于目标平台可能没有的特性的代码。对于本例，我们将添加一些取决于目标平台是否具有 `log` 和 `exp` 函数的代码。当然，几乎每个平台都具有这些功能，但本教程假设它们并不常见。

如果平台具有 `log` 和 `exp`，那么我们将使用它们来计算 `mysqrt` 函数中的平方根。我们首先使用 `MathFunctions/CMakeLists.txt` 中的 `CheckSymbolExists` 模块测试这些功能的可用性。在某些平台上，我们将需要链接到 m 库。如果最初没有找到 `log` 和 `exp`，则请求 m 库，然后重试。

```cmake
include(CheckSymbolExists)
check_symbol_exists(log "math.h" HAVE_LOG)
check_symbol_exists(exp "math.h" HAVE_EXP)
if(NOT (HAVE_LOG AND HAVE_EXP))
  unset(HAVE_LOG CACHE)
  unset(HAVE_EXP CACHE)
  set(CMAKE_REQUIRED_LIBRARIES "m")
  check_symbol_exists(log "math.h" HAVE_LOG)
  check_symbol_exists(exp "math.h" HAVE_EXP)
  if(HAVE_LOG AND HAVE_EXP)
    target_link_libraries(MathFunctions PRIVATE m)
  endif()
endif()
```

如果可用，使用 `target_compile_definitions()` 指定 `HAVE_LOG` 和 `HAVE_EXP` 作为 `PRIVATE` 编译定义。

```cmake
if(HAVE_LOG AND HAVE_EXP)
  target_compile_definitions(MathFunctions
                             PRIVATE "HAVE_LOG" "HAVE_EXP")
endif()
```

如果 `log` 和 `exp` 在系统上可用，那么我们将使用它们来计算 `mysqrt` 函数中的平方根。将以下代码添加到 `MathFunctions/mysqrt.cxx` 中的 `mysqrt` 函数中（返回结果前不要忘记 `#endif`）：

```cpp
#if defined(HAVE_LOG) && defined(HAVE_EXP)
  double result = exp(log(x) * 0.5);
  std::cout << "Computing sqrt of " << x << " to be " << result
            << " using log and exp" << std::endl;
#else
  double result = x;
```

我们还需要修改 `mysqrt.cxx` 以 include `cmath`。

```cpp
#include <cmath>
```

运行 `cmake` 可执行文件或 `cmake-gui` 来配置项目，然后用你选择的构建工具构建它，并运行 Tutorial 可执行文件。

现在哪个函数给出了更好的结果，sqrt 还是 mysqrt？

## 4.6 添加自定义命令和生成的文件（第六步）

假设，出于本教程的目的，我们决定不再使用平台的 `log` 和 `exp` 函数，而是希望生成一个可在 `mysqrt` 函数中使用的预计算值表。在本节中，我们将在构建过程中创建表，然后将该表编译到我们的应用程序中。

首先，让我们在 `MathFunctions/CMakeLists.txt` 中删除对 `log` 和 `exp` 函数的检查。然后从 `mysqrt.cxx` 中删除对 `HAVE_LOG` 和 `HAVE_EXP` 的检查。同时，我们可以删除 `#include <cmath>`。

在 `MathFunctions` 子目录中，提供了一个名为 `MakeTable.cxx` 的新源文件来生成这张表。

我们可以看到该表是作为有效的 C++ 代码生成的，并且输出文件名已作为参数传递。

下一步是将适当的命令添加到 `MathFunctions/CMakeLists.txt` 文件中，以构建 MakeTable 可执行文件，然后在构建过程中运行它。我们需要一些命令来完成此操作。

首先，在 `MathFunctions/CMakeLists.txt` 的顶部，添加 MakeTable 的可执行文件，就像添加任何其他可执行文件一样。

```cmake
add_executable(MakeTable MakeTable.cxx)
```

然后，我们添加一个自定义命令，该命令指定如何通过运行 MakeTable 来产生 `Table.h`。

```cmake
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
  COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
  DEPENDS MakeTable
  )
```

接下来，我们必须让 CMake 知道 `mysqrt.cxx` 取决于生成的文件 `Table.h`。这是通过将生成的 `Table.h` 添加到库 MathFunctions 的源列表中来完成的。

```cmake
add_library(MathFunctions
            mysqrt.cxx
            ${CMAKE_CURRENT_BINARY_DIR}/Table.h
            )
```

我们还必须将当前的二进制目录添加到 include 目录列表中，以便 `mysqrt.cxx` 可以找到并包含 `Table.h`。

```cmake
target_include_directories(MathFunctions
          INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
          PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
          )
```

现在，让我们使用生成的表。首先，修改 `mysqrt.cxx` 以包含 `Table.h`。接下来，我们可以重写 `mysqrt` 函数以使用该表：

```cpp
double mysqrt(double x)
{
  if (x <= 0) {
    return 0;
  }

  // 使用这张表来帮助寻找一个初始值
  double result = x;
  if (x >= 1 && x < 10) {
    std::cout << "Use the table to help find an initial value " << std::endl;
    result = sqrtTable[static_cast<int>(x)];
  }

  // 做十次迭代
  for (int i = 0; i < 10; ++i) {
    if (result <= 0) {
      result = 0.1;
    }
    double delta = x - (result * result);
    result = result + 0.5 * delta / result;
    std::cout << "Computing sqrt of " << x << " to be " << result << std::endl;
  }

  return result;
}
```

运行 `cmake` 可执行文件或 `cmake-gui` 来配置项目，然后用你选择的构建工具构建它。

构建此项目时，它将首先构建 `MakeTable` 可执行文件。然后它将运行 `MakeTable` 生成 `Table.h`。最后，它将编译包括 `Table.h` 的 `mysqrt.cxx`，以生成 MathFunctions 库。

运行 Tutorial 可执行文件，并验证它是否正在使用该表。

## 4.7 构建安装程序（第七步）

接下来，假设我们想将项目分发给其他人，以便他们可以使用它。我们希望在各种平台上提供二进制和源代码分发。这与之前在安装和测试（第四步）中进行的安装有些不同，在安装和测试中，我们安装的是根据源代码构建的二进制文件。在此示例中，我们将构建支持二进制安装和程序包管理功能的安装程序包。为此，我们将使用 CPack 创建特定于平台的安装程序。具体来说，我们需要在顶级 `CMakeLists.txt` 文件的底部添加几行。

```cmake
include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${Tutorial_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Tutorial_VERSION_MINOR}")
include(CPack)
```

就是这样。我们首先 include `InstallRequiredSystemLibraries`。该模块将包括项目当前平台所需的任何运行时库。接下来，我们将一些 CPack 变量设置为存储该项目的许可证和版本信息的位置。版本信息是在本教程的前面设置的，并且 `license.txt` 已包含在此步骤的顶级源目录中。

最后，我们 include `CPack module`，该模块将使用这些变量和当前系统的其他一些属性来设置安装程序。

下一步是以通常的方式构建项目，然后运行 `cpack` 可执行文件。要构建二进制发行版，请从二进制目录运行：

```powershell
cpack
```

要指定生成器，请使用 `-G` 选项。对于多配置构建，请使用 `-C` 指定配置。例如：

```powershell
cpack -G ZIP -C Debug
```

要创建一个源分发版，你可以输入：

```powershell
cpack --config CPackSourceConfig.cmake
```

或者，运行 `make package` 或在 IDE 中右键单击 `Package` 目标和 `Build Project`。

运行在二进制目录中找到的安装程序。然后运行已安装的可执行文件，并验证其是否工作正常。

## 4.8 添加对 Dashboard 的支持（第八步）

添加将测试结果提交到 Dashboard 的支持非常简单。我们已经在“测试支持”中为我们的项目定义了许多测试。现在，我们只需要运行这些测试并将其提交到 Dashboard 即可。为了包含对 Dashboard 的支持，我们在顶层 `CMakeLists.txt` 中包含了 `CTest` 模块。

将这个：

```cmake
# 启用测试
enable_testing()
```

替换为：

```cmake
# 启用 Dashboard 脚本
include(CTest)
```

`CTest` 模块将自动调用 `enable_testing()`，因此我们可以将其从 CMake 文件中删除。

我们还需要在顶级目录中创建一个 `CTestConfig.cmake` 文件，在该目录中我们可以指定项目的名称以及提交 Dashboard 的位置。

```cmake
set(CTEST_PROJECT_NAME "CMakeTutorial")
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=CMakeTutorial")
set(CTEST_DROP_SITE_CDASH TRUE)
```

`ctest` 可执行文件将在运行时读入此文件。要创建一个简单的 Dashboard，你可以运行 `cmake` 可执行文件或 `cmake-gui` 来配置项目，但现在还不要构建它，而是将目录更改为二进制目录，然后运行：

```powershell
ctest [-VV] -D Experimental
```

请记住，对于多配置生成器（例如 Visual Studio），必须指定配置类型：

```powershell
ctest [-VV] -C Debug -D Experimental
```

或者，从 IDE 中构建 `Experimental` 目标。

`ctest` 可执行文件将构建和测试项目，并将结果提交到 Kitware 的公共 Dashboard：[https://my.cdash.org/index.php?project=CMakeTutorial](https://link.zhihu.com/?target=https%3A//my.cdash.org/index.php%3Fproject%3DCMakeTutorial)。

## 4.9 混合静态库和动态库（第九步）

在本节中，我们将展示如何使用 `BUILD_SHARED_LIBS` 变量来控制 `add_library()` 的默认行为，并允许控制如何构建没有显式类型（`STATIC`、`SHARED`、`MODULE` 或 `OBJECT`）的库。

为此，我们需要将 `BUILD_SHARED_LIBS` 添加到顶级 `CMakeLists.txt` 中。我们使用 `option()` 命令，因为它允许用户自由选择该值应为 ON 还是 OFF。

接下来，我们将重构 MathFunctions 使其成为使用 `mysqrt` 或 `sqrt` 封装的真实库，而不是要求调用代码执行此逻辑。这也意味着 `USE_MYMATH` 将不会控制构建 MathFunction，而是将控制此库的行为。

```cmake
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和版本
project(Tutorial VERSION 1.0)

# 指定 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 控制静态库和动态库的构建位置，以便在 Windows 上
# 我们无需修改运行可执行文件的路径
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# 配置一个仅用于传递版本号的头文件
configure_file(TutorialConfig.h.in TutorialConfig.h)

# 添加 MathFunctions 库
add_subdirectory(MathFunctions)

# 添加可执行文件
add_executable(Tutorial tutorial.cxx)
target_link_libraries(Tutorial PUBLIC MathFunctions)
```

现在我们已经使 MathFunctions 始终被使用，我们将需要更新该库的逻辑。因此，在 `MathFunctions/CMakeLists.txt` 中，我们需要创建一个 SqrtLibrary，当启用 `USE_MYMATH` 时将有条件地构建和安装该 SqrtLibrary。现在，由于这是一个教程，我们将明确要求 SqrtLibrary 是静态构建的。

最终结果是 `MathFunctions/CMakeLists.txt` 看起来应该像这样：

```cmake
# 添加库
add_library(MathFunctions MathFunctions.cxx)

# 设置任何连接到我们这个库的人必须包含的目录
# 但我们自己不需要包含
target_include_directories(MathFunctions
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           )

# 我们是否要使用自己的数学库
option(USE_MYMATH "Use tutorial provided math implementation" ON)
if(USE_MYMATH)

  target_compile_definitions(MathFunctions PRIVATE "USE_MYMATH")

  # 首先我们添加可执行文件来生成表
  add_executable(MakeTable MakeTable.cxx)

  # 添加命令生成源码
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    DEPENDS MakeTable
    )

  # 做 sqrt 的库
  add_library(SqrtLibrary STATIC
              mysqrt.cxx
              ${CMAKE_CURRENT_BINARY_DIR}/Table.h
              )

  # 声明我们依靠二进制目录找到 Table.h
  target_include_directories(SqrtLibrary PRIVATE
                             ${CMAKE_CURRENT_BINARY_DIR}
                             )

  target_link_libraries(MathFunctions PRIVATE SqrtLibrary)
endif()

# 定义说明在Windows上构建时使用
# declspec（dllexport）的符号
target_compile_definitions(MathFunctions PRIVATE "EXPORTING_MYMATH")

# 安装规则
set(installable_libs MathFunctions)
if(TARGET SqrtLibrary)
  list(APPEND installable_libs SqrtLibrary)
endif()
install(TARGETS ${installable_libs} DESTINATION lib)
install(FILES MathFunctions.h DESTINATION include)
```

接下来，更新 MathFunctions/mysqrt.cxx 以使用 mathfunctions 和 detail 命名空间：

```cpp
#include <iostream>

#include "MathFunctions.h"

// include the generated table
#include "Table.h"

namespace mathfunctions {
namespace detail {
// a hack square root calculation using simple operations
double mysqrt(double x)
{
  if (x <= 0) {
    return 0;
  }

  // use the table to help find an initial value
  double result = x;
  if (x >= 1 && x < 10) {
    std::cout << "Use the table to help find an initial value " << std::endl;
    result = sqrtTable[static_cast<int>(x)];
  }

  // do ten iterations
  for (int i = 0; i < 10; ++i) {
    if (result <= 0) {
      result = 0.1;
    }
    double delta = x - (result * result);
    result = result + 0.5 * delta / result;
    std::cout << "Computing sqrt of " << x << " to be " << result << std::endl;
  }

  return result;
}
}
}
```

我们还需要在 tutorial.cxx 中进行一些更改，以使其不再使用 USE_MYMATH：

1.  始终 include `MathFunctions.h`

2.  始终使用 `mathfunctions::sqrt`

3.  不 include `cmath`

最后，更新 MathFunctions/MathFunctions.h 以使用 dll 导出定义：

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

此时，如果您构建了所有内容，则可能会注意到链接失败，因为我们将没有位置独立代码的静态库与具有位置独立代码的库组合在一起。解决方案是无论构建类型如何，都将 SqrtLibrary 的 `POSITION_INDEPENDENT_CODE` 目标属性显式设置为 True。

```cmake
# state that SqrtLibrary need PIC when the default is shared libraries
  set_target_properties(SqrtLibrary PROPERTIES
                        POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
                        )

  target_link_libraries(MathFunctions PRIVATE SqrtLibrary)
```

**练习**：我们修改了 `MathFunctions.h` 以使用 dll 导出定义。借助 CMake 文档，您能找到一个帮助模块来简化此过程吗？

## 4.10 添加生成器表达式（第十步）

在构建系统生成期间会评估生成器表达式，以生成特定于每个构建配置的信息。

在许多目标属性的上下文中允许使用生成器表达式，例如 `LINK_LIBRARIES`，`INCLUDE_DIRECTORIES`，`COMPLIE_DEFINITIONS` 等。在使用命令填充这些属性（例如 `target_link_libraries()`，`target_include_directories()`，`target_compile_definitions()` 等）时，也可以使用它们。

生成器表达式可用于启用条件链接，编译时使用的条件定义，条件包含目录等。条件可以基于构建配置，目标属性，平台信息或任何其他可查询的信息。

生成器表达式有许多不同的类型，包括逻辑表达式、信息表达式和输出表达式。

逻辑表达式用于创建条件输出。基本表达式是 0 和 1 表达式。一个 `$<0:...>` 生成空字符串，而 `<1:...>` 生成“...”。它们也可以被嵌套。

生成器表达式的常见用法是有条件地添加编译器标志，例如用于语言级别或警告的标志。一个不错的模式是将该信息与一个接口目标相关联，以允许该信息传播。让我们从构造一个 `INTERFACE` 目标并指定所需的 C++ 标准级别 11 开始，而不是使用 `CMAKE_CXX_STANDARD`。

所以下面的代码：

```cmake
# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```

将被替换为：

```cmake
add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_11)
```

接下来，我们为我们的项目添加所需的编译器警告标志。由于警告标志根据编译器的不同而不同，因此我们使用 `COMPILE_LANG_AND_ID` 生成器表达式来控制在给定一种语言和一组编译器 ID 的情况下应该应用的标志，如下所示：

```cmake
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")
target_compile_options(tutorial_compiler_flags INTERFACE
  "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
  "$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>"
)
```

查看此内容，我们看到警告标志封装在 `BUILD_INTERFACE` 条件内。这样做是为了使我们已安装项目的使用者不会继承我们的警告标志。

**练习**：修改 `MathFunctions/CMakeLists.txt`，以便所有目标都具有对 `tutorial_compiler_flags` 的 `target_link_libraries()` 调用。

## 4.11 添加导出配置（第十一步）

在本教程的安装和测试（第四步）中，我们添加了 CMake 的功能，以安装项目的库和头文件。在构建安装程序（第七步）期间，我们添加了打包此信息的功能，以便可以将其分发给其他人。

下一步是添加必要的信息，以便其他 CMake 项目可以使用我们的项目，无论是从构建目录，本地安装还是打包时。

第一步是更新我们的 `install(TARGETS)` 命令，不仅要指定 `DESTINATION`，还要指定 `EXPORT`。 `EXPORT` 关键字生成并安装一个 CMake 文件，该文件包含用于从安装树中导入 `install` 命令中列出的所有目标的代码。因此，让我们继续，通过更新 `MathFunctions/CMakeLists.txt` 中的安装命令，显式导出 MathFunctions 库，如下所示：

```cmake
set(installable_libs MathFunctions tutorial_compiler_flags)
if(TARGET SqrtLibrary)
  list(APPEND installable_libs SqrtLibrary)
endif()
install(TARGETS ${installable_libs}
        DESTINATION lib
        EXPORT MathFunctionsTargets)
install(FILES MathFunctions.h DESTINATION include)
```

现在我们已经导出了 MathFunctions，我们还需要显式安装生成的 `MathFunctionsTargets.cmake` 文件。这是通过将以下内容添加到顶级 `CMakeLists.txt` 的底部来完成的：

```cmake
install(EXPORT MathFunctionsTargets
  FILE MathFunctionsTargets.cmake
  DESTINATION lib/cmake/MathFunctions
)
```

此时，您应该尝试运行 CMake。如果一切设置正确，您将看到 CMake 生成如下错误：

```powershell
Target "MathFunctions" INTERFACE_INCLUDE_DIRECTORIES property contains
path:

  "/Users/robert/Documents/CMakeClass/Tutorial/Step11/MathFunctions"

which is prefixed in the source directory.
```

CMake 试图说明的是，在生成导出信息的过程中，它将导出一个本质上与当前机器相关联的路径，该路径在其他机器上无效。解决这个问题的方法是更新 MathFunctions 的 `target_include_directories()`，以理解在构建目录和安装/包中使用它时需要不同的接口位置。这意味着将 MathFunctions 的 `target_include_directories()` 调用转换成如下所示：

```cmake
target_include_directories(MathFunctions
                           INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )
```

更新后，我们可以重新运行 CMake 并确认它不再发出警告。

至此，我们已经正确地包装了 CMake 所需的目标信息，但仍然需要生成 `MathFunctionsConfig.cmake`，以便 CMake `find_package()` 命令可以找到我们的项目。因此，我们继续将新文件添加到名为 `Config.cmake.in` 的项目的顶层，其内容如下：

```cmake
@PACKAGE_INIT@

include ( "${CMAKE_CURRENT_LIST_DIR}/MathFunctionsTargets.cmake" )
```

然后，要正确配置和安装该文件，请将以下内容添加到顶级 `CMakeLists.txt` 的底部：

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

至此，我们为项目生成了可重定位的 CMake 配置，可在安装或打包项目后使用它。如果我们也希望从构建目录中使用我们的项目，则只需将以下内容添加到顶级 `CMakeLists.txt` 的底部：

```cmake
export(EXPORT MathFunctionsTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/MathFunctionsTargets.cmake"
)
```

通过此导出调用，我们现在生成一个 `Targets.cmake`，允许在构建目录中配置的 `MathFunctionsConfig.cmake` 由其他项目使用，而无需安装它。

## 4.12 打包调试和发布（第十二步）

**注意**：此示例对单配置生成器有效，不适用于多配置生成器（例如 Visual Studio）。

默认情况下，CMake 的模型是构建目录仅包含一个配置，可以是 Debug、Release、MinSizeRel 或 RelWithDebInfo。但是，可以将 CPack 设置为捆绑多个构建目录，并构建一个包含同一项目的多个配置的软件包。

首先，我们要确保调试版本和发行版本对要安装的可执行文件和库使用不同的名称。让我们将 d 用作调试可执行文件和库的后缀。

在顶级 `CMakeLists.txt` 文件的开头附近设置 `CMAKE_DEBUG_POSTFIX`：

```cmake
set(CMAKE_DEBUG_POSTFIX d)

add_library(tutorial_compiler_flags INTERFACE)
```

以及本教程可执行文件上的 `DEBUG_POSTFIX` 属性：

```cmake
add_executable(Tutorial tutorial.cxx)
set_target_properties(Tutorial PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

target_link_libraries(Tutorial PUBLIC MathFunctions)
```

让我们还将版本编号添加到 MathFunctions 库中。在 `MathFunctions/CMakeLists.txt` 中，设置 `VERSION` 和 `SOVERSION` 属性：

```cmake
set_property(TARGET MathFunctions PROPERTY VERSION "1.0.0")
set_property(TARGET MathFunctions PROPERTY SOVERSION "1")
```

在 Step12 目录中，创建调试和发布子目录。布局将如下所示：

```text
- Step12
   - debug
   - release
```

现在我们需要设置调试和发布版本。我们可以使用 `CMAKE_BUILD_TYPE` 来设置配置类型：

```powershell
cd debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
cd ../release
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

既然调试和发布版本均已完成，我们就可以使用自定义配置文件将两个版本打包到一个版本中。在 Step12 目录中，创建一个名为 `MultiCPackConfig.cmake` 的文件。在此文件中，首先包括由 cmake 可执行文件创建的默认配置文件。

接下来，使用 `CPACK_INSTALL_CMAKE_PROJECTS` 变量来指定要安装的项目。在这种情况下，我们要同时安装调试和发行版。

```cmake
include("release/CPackConfig.cmake")

set(CPACK_INSTALL_CMAKE_PROJECTS
    "debug;Tutorial;ALL;/"
    "release;Tutorial;ALL;/"
    )
```

在 Step12 目录中，运行 `cpack`，并使用 `config` 选项指定我们的自定义配置文件：

```powershell
cpack --config MultiCPackConfig.cmake
```









