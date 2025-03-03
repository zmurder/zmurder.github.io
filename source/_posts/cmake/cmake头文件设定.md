# 简介

在使用cmake时，编写的CMakeLists.txt文件中需要包含项目需要的头文件，但是我只到有两种方式可以设置头文件的路径

* target_include_directories
* include_directories

一直没有明白这两个之间的关系，一般都偷懒使用了`include_directories`,也不是不行，就是有时候编译大的项目会非常的慢。

因此再了解了一下头文件的路径设置方式。

为了更清晰地展示 `target_include_directories` 和 `include_directories` 在实际项目中的应用差异，下面通过一个稍微复杂一点的例子来说明。假设我们有一个包含多个模块（库和可执行文件）的C++项目，每个模块都有其特定的头文件依赖关系。

## 项目结构

```
my_project/
├── CMakeLists.txt
├── src/
│   ├── main.cpp
│   └── utils.cpp
├── include/
│   └── utils.h
├── lib/
│   ├── mathlib/
│   │   ├── CMakeLists.txt
│   │   ├── src/
│   │   │   └── mathfuncs.cpp
│   │   └── include/
│   │       └── mathfuncs.h
│   └── logger/
│       ├── CMakeLists.txt
│       ├── src/
│       │   └── logger.cpp
│       └── include/
│           └── logger.h
```

## 使用 `include_directories` 的方式

首先，使用全局的 `include_directories` 方式来组织这个项目。这种方式虽然简单，但不够灵活，可能会导致不必要的包含路径被添加到不相关的目标中。

### 根目录下的 `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 全局设置包含路径
include_directories(include)
include_directories(lib/mathlib/include)
include_directories(lib/logger/include)

add_subdirectory(lib/mathlib)
add_subdirectory(lib/logger)

add_executable(myapp src/main.cpp src/utils.cpp)
target_link_libraries(myapp MathLib Logger)
```



`lib/mathlib/CMakeLists.txt`

```cmake
add_library(MathLib src/mathfuncs.cpp)
```



`lib/logger/CMakeLists.txt`

```cmake
add_library(Logger src/logger.cpp)
```

在这个例子中，所有目标都会获得 `include`, `lib/mathlib/include`, 和 `lib/logger/include` 这三个目录作为头文件搜索路径。这种方式虽然简单，但如果某个目标不需要某些包含路径，则会引入不必要的依赖。

## 使用 `target_include_directories` 的方式

接下来，我们将使用 `target_include_directories` 来精确控制每个目标的包含路径，从而提高项目的模块化程度和可维护性。

### 根目录下的 `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

add_subdirectory(lib/mathlib)
add_subdirectory(lib/logger)

add_executable(myapp src/main.cpp src/utils.cpp)
target_include_directories(myapp PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(myapp MathLib Logger)
```



`lib/mathlib/CMakeLists.txt`

```cmake
add_library(MathLib src/mathfuncs.cpp)
target_include_directories(MathLib 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
```



`lib/logger/CMakeLists.txt`

```cmake
add_library(Logger src/logger.cpp)
target_include_directories(Logger 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
```

在这个改进后的版本中：

- **MathLib** 和 **Logger** 库都明确指定了它们需要的包含路径，并且这些路径是以 `PUBLIC` 的形式指定的。这意味着任何链接了这些库的目标也会自动获得这些包含路径。
  
- **myapp** 只需要知道它直接使用的头文件位置 (`${CMAKE_SOURCE_DIR}/include`)，以及通过链接 `MathLib` 和 `Logger` 获得的间接包含路径。

# 比较与总结

- **灵活性**：使用 `target_include_directories` 提供了更高的灵活性，可以针对每个目标精确地控制其包含路径。这对于大型项目尤为重要，因为它减少了不必要的依赖，避免了潜在的冲突。

- **维护性**：由于每个目标的依赖关系更加明确，使用 `target_include_directories` 可以使项目更容易维护。当项目规模扩大时，这种方法有助于保持代码的清晰度和模块化设计。

- **性能**：在构建过程中，减少不必要的包含路径可以加快编译速度，尤其是在大型项目中。

通过上述例子可以看出，尽管 `include_directories` 对于快速配置小项目非常方便，但对于更大、更复杂的项目，推荐使用 `target_include_directories` 来更好地管理依赖关系和模块化设计。这样不仅提高了代码的可读性和维护性，还能有效避免因不当的包含路径设置而导致的问题。