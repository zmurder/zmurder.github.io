# 1 简介

CUDA 提供了许多环境变量来配置和优化 CUDA 运行时和驱动程序的行为。这些环境变量通常用来控制CUDA的性能、调试和设备行为等。

下表列出了 CUDA 环境变量。 与多进程服务相关的环境变量记录在 GPU 部署和管理指南的多进程服务部分。Table 18. CUDA Environment Variables

| Variable                                                 | Values                                                       | Description                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Device Enumeration and Properties**                    |                                                              |                                                              |
| CUDA_VISIBLE_DEVICES                                     | A comma-separated sequence of GPU identifiers  MIG support: MIG-<GPU-UUID>/<GPU instance ID>/<compute instance ID> | GPU identifiers are given as integer indices or as UUID strings. GPU UUID  strings should follow the same format as given by nvidia-smi, such as  GPU-8932f937-d72c-4106-c12f-20bd9faed9f6. However, for convenience,  abbreviated forms are allowed; simply specify enough digits from the  beginning of the GPU UUID to uniquely identify that GPU in the target  system. For example, CUDA_VISIBLE_DEVICES=GPU-8932f937 may be a valid  way to refer to the above GPU UUID, assuming no other GPU in the system  shares this prefix.  Only the devices whose index is present in  the sequence are visible to CUDA applications and they are enumerated in the order of the sequence. If one of the indices is invalid, only the  devices whose index precedes the invalid index are visible to CUDA  applications. For example, setting CUDA_VISIBLE_DEVICES to 2,1 causes  device 0 to be invisible and device 2 to be enumerated before device 1.  Setting CUDA_VISIBLE_DEVICES to 0,2,-1,1 causes devices 0 and 2 to be  visible and device 1 to be invisible.  MIG format starts with MIG  keyword and GPU UUID should follow the same format as given by  nvidia-smi. For example,  MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2. Only single MIG  instance enumeration is supported. |
| CUDA_MANAGED_FORCE_DEVICE_ALLOC                          | 0 or 1 (default is 0)                                        | Forces the driver to place all managed allocations in device memory. |
| CUDA_DEVICE_ORDER                                        | FASTEST_FIRST, PCI_BUS_ID, (default is FASTEST_FIRST)        | FASTEST_FIRST causes CUDA to enumerate the available devices in fastest to slowest  order using a simple heuristic. PCI_BUS_ID orders devices by PCI bus ID  in ascending order. |
| **Compilation**                                          |                                                              |                                                              |
| CUDA_CACHE_DISABLE                                       | 0 or 1 (default is 0)                                        | Disables caching (when set to 1) or enables caching (when set to 0) for  just-in-time-compilation. When disabled, no binary code is added to or  retrieved from the cache. |
| CUDA_CACHE_PATH                                          | filepath                                                     | Specifies the folder where the just-in-time compiler caches binary codes; the default values are:on Windows, %APPDATA%\NVIDIA\ComputeCacheon Linux, ~/.nv/ComputeCache |
| CUDA_CACHE_MAXSIZE                                       | integer (default is 268435456 (256 MiB) and maximum is 4294967296 (4 GiB)) | Specifies the size in bytes of the cache used by the just-in-time compiler.  Binary codes whose size exceeds the cache size are not cached. Older  binary codes are evicted from the cache to make room for newer binary  codes if needed. |
| CUDA_FORCE_PTX_JIT                                       | 0 or 1 (default is 0)                                        | When set to 1, forces the device driver to ignore any binary code embedded in an application (see [Application Compatibility](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese/blob/main/附录M_CUDA环境变量/index.html#application-compatibility)) and to just-in-time compile embedded PTX code instead. If a kernel does not have embedded PTX code, it will fail to load. This environment  variable can be used to validate that PTX code is embedded in an  application and that its just-in-time compilation works as expected to  guarantee application forward compatibility with future architectures  (see [Just-in-Time Compilation](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese/blob/main/附录M_CUDA环境变量/index.html#just-in-time-compilation)). |
| CUDA_DISABLE_PTX_JIT                                     | 0 or 1 (default is 0)                                        | When set to 1, disables the just-in-time compilation of embedded PTX code  and use the compatible binary code embedded in an application (see [Application Compatibility](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese/blob/main/附录M_CUDA环境变量/index.html#application-compatibility)). If a kernel does not have embedded binary code or the embedded binary  was compiled for an incompatible architecture, then it will fail to  load. This environment variable can be used to validate that an  application has the compatible SASS code generated for each kernel.(see [Binary Compatibility](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese/blob/main/附录M_CUDA环境变量/index.html#binary-compatibility)). |
| **Execution**                                            |                                                              |                                                              |
| CUDA_LAUNCH_BLOCKING                                     | 0 or 1 (default is 0)                                        | Disables (when set to 1) or enables (when set to 0) asynchronous kernel launches. |
| CUDA_DEVICE_MAX_CONNECTIONS                              | 1 to 32 (default is 8)                                       | Sets the number of compute and copy engine concurrent connections (work  queues) from the host to each device of compute capability 3.5 and  above. |
| CUDA_AUTO_BOOST                                          | 0 or 1                                                       | Overrides the autoboost behavior set by the –auto-boost-default option of  nvidia-smi. If an application requests via this environment variable a  behavior that is different from nvidia-smi’s, its request is honored if  there is no other application currently running on the same GPU that  successfully requested a different behavior, otherwise it is ignored. |
| **cuda-gdb (on Linux platform)**                         |                                                              |                                                              |
| CUDA_DEVICE_WAITS_ON_EXCEPTION                           | 0 or 1 (default is 0)                                        | When set to 1, a CUDA application will halt when a device exception occurs,  allowing a debugger to be attached for further debugging. |
| **MPS service (on Linux platform)**                      |                                                              |                                                              |
| CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT | Percentage value (between 0 – 100, default is 0)             | Devices of compute capability 8.x allow, a portion of L2 cache to be set-aside  for persisting data accesses to global memory. When using CUDA MPS  service, the set-aside size can only be controlled using this  environment variable, before starting CUDA MPS control daemon. I.e., the environment variable should be set before running the command nvidia-cuda-mps-control -d. |

下面是中文的一些说明

## 常见的 CUDA 环境变量

### **CUDA_DEVICE_MAX_CONNECTIONS**

- **描述**：设置一个设备允许的最大连接数。适用于多进程应用程序，当多个进程需要访问同一 GPU 时，此变量可以控制并发连接数。
- 这个我目前常用到需要设置的一个环境变量，默认是8，我一般设置为最大的32.以便有更多的stream并发。这在我的文章 [6-2 并发内核执行](https://zmurder.github.io/CUDA/CUDA%20C%E7%BC%96%E7%A8%8B%E6%9D%83%E5%A8%81%E6%8C%87%E5%8D%97%E7%AC%94%E8%AE%B0/6-2%20%E5%B9%B6%E5%8F%91%E5%86%85%E6%A0%B8%E6%89%A7%E8%A1%8C/?highlight=cuda_device_max_connections)中也有描述
- **默认值**：8
- **示例**：
  ```sh
  export CUDA_DEVICE_MAX_CONNECTIONS=16
  ```

### **CUDA_VISIBLE_DEVICES**

- **描述**：控制CUDA程序可见的 GPU 设备。指定要使用的 GPU 设备的列表，其他设备将被忽略。
- **示例**：
  ```sh
  export CUDA_VISIBLE_DEVICES=0,1
  ```
- **功能**：只允许 CUDA 程序访问 GPU 设备 0 和 1。

### **CUDA_LAUNCH_BLOCKING**

- **描述**：控制CUDA内核的同步行为。设置为1时，CUDA调用是同步的，这对于调试非常有用，因为它确保所有内核调用和内存拷贝操作完成后才继续执行。
- **默认值**：0（异步）
- **示例**：
  ```sh
  export CUDA_LAUNCH_BLOCKING=1
  ```

### **CUDA_MPS_PIPE_DIRECTORY**

- **描述**：指定 CUDA MPS（Multi-Process Service）管道目录的位置。CUDA MPS 允许多个进程共享 GPU 上的上下文。
- **默认值**：`/tmp/nvidia-mps`
- **示例**：
  ```sh
  export CUDA_MPS_PIPE_DIRECTORY=/var/tmp/nvidia-mps
  ```

### **CUDA_MPS_LOG_DIRECTORY**

- **描述**：指定 CUDA MPS 的日志文件目录。用于存储 MPS 运行时生成的日志文件。
- **默认值**：`/tmp/nvidia-mps`
- **示例**：
  ```sh
  export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
  ```

### **CUDA_DEVICE_WARP_SIZE**

- **描述**：设置 CUDA 设备的 warp 大小。大部分现代 CUDA 设备的 warp 大小是 32，但这个变量可以在某些调试场景中使用。
- **示例**：
  ```sh
  export CUDA_DEVICE_WARP_SIZE=32
  ```

### **CUDA_CACHE_MAXSIZE**

- **描述**：设置 CUDA 内核和模块缓存的最大大小。此变量控制缓存的大小以管理 GPU 的内存使用。
- **默认值**：`0`（不限制）
- **示例**：
  ```sh
  export CUDA_CACHE_MAXSIZE=1024
  ```

### **CUDA_CACHE_DISABLE**

- **描述**：禁用 CUDA 的内核和模块缓存。对调试和性能优化有帮助。
- **默认值**：0（启用缓存）
- **示例**：
  ```sh
  export CUDA_CACHE_DISABLE=1
  ```

### **CUDA_FPE_CONTROL**

- **描述**：控制 CUDA 运行时浮点异常（FPE）的处理。此变量对调试浮点错误非常有用。
- **示例**：
  ```sh
  export CUDA_FPE_CONTROL=1
  ```

### **CUDA_DEVICE_DEBUG_SYNC**

- **描述**：启用 CUDA 设备的调试同步。这对于调试应用程序中出现的异常行为非常有用。
- **默认值**：0（禁用）
- **示例**：
  ```sh
  export CUDA_DEVICE_DEBUG_SYNC=1
  ```

### 配置示例

以下是一个配置示例，将这些环境变量添加到 `.bashrc` 或 `.bash_profile` 文件中，以便在每次启动终端时自动设置这些变量：

```sh
# 控制 CUDA 设备的最大连接数
export CUDA_DEVICE_MAX_CONNECTIONS=16

# 指定可见的 GPU 设备
export CUDA_VISIBLE_DEVICES=0,1

# 启用同步内核调用
export CUDA_LAUNCH_BLOCKING=1

# 设置 CUDA MPS 管道目录
export CUDA_MPS_PIPE_DIRECTORY=/var/tmp/nvidia-mps

# 设置 CUDA MPS 日志目录
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps

# 设置 CUDA 设备 warp 大小
export CUDA_DEVICE_WARP_SIZE=32

# 设置 CUDA 缓存最大大小
export CUDA_CACHE_MAXSIZE=1024

# 禁用 CUDA 缓存
export CUDA_CACHE_DISABLE=1

# 启用 CUDA 浮点异常控制
export CUDA_FPE_CONTROL=1

# 启用 CUDA 设备调试同步
export CUDA_DEVICE_DEBUG_SYNC=1
```

在保存并重新加载配置文件后（使用 `source ~/.bashrc`），这些环境变量将生效，从而调整 CUDA 的运行时行为。