# 3 cmake使用规则

## 3.1 从命令行定义全局变量

在执行 cmake 指令的时候，可以定义任意多个全局变量。这些全局变量可以直接在CMakeLists.txt 中被使用。这是一项很方便的功能。

例如

```shell
$cmake  ..   -DCONFIG=Debug   -DSSE=True 
```

这条指令定义了两个全局变量：`CONFIG` 和 `SSE`，其值分别是"Debug"和"True"。 不要被这两个变量前面的-D 所迷惑。那只是用来告诉 cmake，要定义变量了。

## 3.2 构建类型

 CMake 为我们提供了四种构建类型： 

* Debug 
* Release 
* MinSizeRel 
* RelWithDebInfo 

如果使用 CMake 生成 Makefile 时，我们需要做一些不同的工作。CMake 中存在一个变量 `CMAKE_BUILD_TYPE` 用于指定构建类型，此变量只用于基于 make 的生成器

例如

```shell
$ cmake  -DCMAKE_BUILD_TYPE=Debug 
```



下面是官方的翻译https://link.zhihu.com/?target=https%3A//cmake.org/cmake/help/latest/guide/tutorial/index.html