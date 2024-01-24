# 10.3 CUDA调试

分为两个部分，内核调试和内存调试。

## 10.3.1 内核调试

内核调试是通过检查一个或多个线程的执行和状态来确定内核的正确性。在CUDA里内核调试有3种主要的方法：cuda-gdb、printf和assert。

### 10.3.1.1 使用cuda-gdb

与gdb类似，cuda有对应的cuda-gdb

**编译**

如果需要使用cuda-gdb调试cuda程序， 编译程序时要添加两个标志到nvcc中：-g和-G

```c
$ nvcc -g -G foo.cu -o foo
```

**调试**

只要使用调试标志编译了应用程序，就能像用gdb那样用cuda-gdb启动一个CUDA应用程序了。给定一个编译和链接应用程序foo，可以通过以下方法将可执行文件传给cuda-gdb：

```C
$ cuda-gdb foo
...
(cuda-gdb)
```

虽然CUDA程序能包含多个主机线程和许多CUDA线程，但是cuda-gdb调试会话一次只处理一个线程。为了能在同一应用程序中调试多个设备线程，cuda-gdb提供了一种功能，即可以指定被检查的上下文（即设备线程）。

**当前焦点**

设备线程属于一个block，而block又属于一个kernel。thread、block、kernel是软件的焦点坐标。设备thread在lane上运行。lane属于warp，warp属于 SM，而 SM  又属于device。 Lane、warp、SM、device是焦点的硬件坐标。软件和硬件坐标可以互换并同时使用，只要它们保持一致即可。

```shell
(cuda-gdb) cuda device sm warp lane block thread
block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0
(cuda-gdb) cuda kernel block thread
kernel 1, block (0,0,0), thread (0,0,0)
(cuda-gdb) cuda kernel
kernel 1
```

**切换焦点**

要切换当前焦点，请使用 cuda 命令，后跟要更改的坐标：

```c
(cuda-gdb) cuda device 0 sm 1 warp 2 lane 3
[Switching focus to CUDA kernel 1, grid 2, block (8,0,0), thread
(67,0,0), device 0, sm 1, warp 2, lane 3]
374 int totalThreads = gridDim.x * blockDim.x;
```

```c
(cuda-gdb) cuda thread (15)
[Switching focus to CUDA kernel 1, grid 2, block (8,0,0), thread
(15,0,0), device 0, sm 1, warp 0, lane 15]
374 int totalThreads = gridDim.x * blockDim.x;
```

```c
(cuda-gdb) cuda block 1 thread 3
[Switching focus to CUDA kernel 1, grid 2, block (1,0,0), thread (3,0,0),
device 0, sm 3, warp 0, lane 3]
374 int totalThreads = gridDim.x * blockDim.
```

**检查CUDA内存**

与gdb一样，cuda-gdb支持检查变量，在堆（即CUDA全局内存）和寄存器中使用print语句：

根据变量类型和用法，变量可以存储在寄存器中，也可以存储在本地、共享、const 或全局内存中。您可以打印任何变量的地址以找出它的存储位置并直接访问关联的内存。

下面的示例显示了如何直接访问共享 int * 类型的变量数组，以查看数组中存储的值。

```c
(cuda-gdb) print &array
$1 = (@shared int (*)[0]) 0x20
(cuda-gdb) print array[0]@4
$2 = {0, 128, 64, 192}
```

您还可以访问索引到起始偏移量的共享内存，以查看存储的值是什么：

```c
(cuda-gdb) print *(@shared int*)0x20
$3 = 0
(cuda-gdb) print *(@shared int*)0x24
$4 = 128
(cuda-gdb) print *(@shared int*)0x28
$5 = 64
```

下面的例子展示了如何访问内核输入参数的起始地址。

```c
(cuda-gdb) print &data
$6 = (const @global void * const @parameter *) 0x10
(cuda-gdb) print *(@global void * const @parameter *) 0x10
$7 = (@global void * const @parameter) 0x110000</>
```

具体的内容可以参考附录的官方文档，还是比较好理解的。

### 10.3.1.2 cuda中的printf

基于CUDA的printf接口，与我们在主机上C/C++研发中习惯使用的一样（甚至有着相同的头文件，stdio.h），这使得我们能直接过渡到基于CUDA的printf中。

```c
__global__ void kernel() {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
printf("Hello from CUDA thread %d\n", tid);
}
```

### 10.3.1.3 CUDA的assert工具

另一个常见的主机错误检查工具是assert。assert能让我们声明某一的条件，在程序正确执行时该条件必须为真。如果assert失败，则应用程序执行有以下两种情况中的一种：1）有assert失败的消息时，立即中止；2）如果在cuda-gdb会话中运行，控制将会传到cuda-gdb，以便可以在assert失败的位置检查应用程序的状态。和printf一样，只有GPU计算能力为2.0及以上时才提供assert功能。它依赖和主机相同的头文件，assert.h。

```c
__global__ void kernel(...) {
int *ptr = NULL;
...
assert(ptr != NULL);
*ptr = ...
...
}
```

与主机的assert一样，通过使用在包含assert.h头文件前定义的NDEBUG预处理器宏编译，可以对代码发行版本禁用assert评估。

## 10.3.2 内存调试

关于cuda-memcheck 还是参考官网吧。后面单独出一篇文章讲解。

# 附录：

* [官方cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
* [官方cuda-memcheck](https://docs.nvidia.com/cuda/archive/11.4.4/cuda-memcheck/index.html#cuda-memcheck-tool-examples) 