# 1 前言

在调查cuda内存问题时可能有以下需求：

1. 跟踪对应内存块的分配位置
2. 跟踪cuda runtime调用参数及结果

# 2 原理

链接器`ld`支持名为`--wrap=symbol`的选项，对`symbol`的任何未定义引用都将解析为`__wrap_symbol`。对`__real_symbol`的任何未定义引用都将解析为符号。这可用于提供系统函数的包装器。包装函数应称为`__wrap_symbol`。如果希望调用系统函数，则应调用`__real_symbol`。

以下是一个简单的示例：

```C
void* __wrap_malloc(size_t c) {  
    printf("malloc called with %zu\n", c);  
    return __real_malloc(c);  
}
```

如果将其他代码与使用`--wrap=malloc`此文件进行链接，那么对`malloc`的所有调用都将调用`__wrap_malloc`函数。在`__wrap_malloc`中调用`__real_malloc`将调用真正的`malloc`函数。

# 3 使用

## 3.1 添加编译选项

在CMakeLists.txt中添加以下选项

```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-linker")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wl,-wrap=cudaMalloc -Wl,-wrap=cudaMallocManaged -Wl,-wrap=cudaMallocHost -Wl,-wrap=cudaHostAlloc  -Wl,-wrap=cudaMemcpy ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wl,-wrap=cudaMemset -Wl,-wrap=cudaMallocAsync -Wl,-wrap=cudaMemcpyAsync -Wl,-wrap=cudaMemsetAsync -Wl,-wrap=cudaLaunchKernel")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wl,-wrap=cudaMallocAsync_ptsz -Wl,-wrap=cudaMemcpyAsync_ptsz -Wl,-wrap=cudaMemsetAsync_ptsz -Wl,-wrap=cudaLaunchKernel_ptsz")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wl,-wrap=cudaFree -Wl,-wrap=cudaFreeHost -Wl,-wrap=cudaFreeAsync -Wl,-wrap=cudaFreeAsync_ptsz ")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-wrap=cudaMalloc -Wl,-wrap=cudaMallocManaged -Wl,-wrap=cudaMallocHost -Wl,-wrap=cudaHostAlloc -Wl,-wrap=cudaMemcpy ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-wrap=cudaMemset -Wl,-wrap=cudaMallocAsync -Wl,-wrap=cudaMemcpyAsync -Wl,-wrap=cudaMemsetAsync -Wl,-wrap=cudaLaunchKernel")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-wrap=cudaMallocAsync_ptsz -Wl,-wrap=cudaMemcpyAsync_ptsz -Wl,-wrap=cudaMemsetAsync_ptsz -Wl,-wrap=cudaLaunchKernel_ptsz")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-wrap=cudaFree -Wl,-wrap=cudaFreeHost -Wl,-wrap=cudaFreeAsync -Wl,-wrap=cudaFreeAsync_ptsz ")
```

## 3.2 添加代码

在任意cpp代码中添加如下内容：

```c++
#include "stdio.h" // 输出信息
#include "unistd.h"  // 获取tid
#include "sys/syscall.h"  // 获取tid

#include "cuda.h"  // cuda runtime
#include "cuda_runtime_api.h"  // cuda runtime


extern "C" {
cudaError_t __wrap_cudaFree ( void* devPtr );
cudaError_t __wrap_cudaFreeHost ( void* ptr );
cudaError_t __wrap_cudaMalloc ( void** devPtr, size_t size );
cudaError_t __wrap_cudaMallocManaged ( void** devPtr, size_t size, unsigned int flags);
cudaError_t __wrap_cudaMallocHost ( void** ptr, size_t size );
cudaError_t __wrap_cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags );
cudaError_t __wrap_cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind );
cudaError_t __wrap_cudaMemset ( void* devPtr, int value, size_t count );
cudaError_t __wrap_cudaFreeAsync ( void* devPtr, cudaStream_t hStream );
cudaError_t __wrap_cudaMallocAsync ( void** devPtr, size_t size, cudaStream_t hStream );
cudaError_t __wrap_cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream );
cudaError_t __wrap_cudaMemsetAsync ( void* devPtr, int value, size_t count, cudaStream_t stream );
cudaError_t __wrap_cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream );
cudaError_t __wrap_cudaFreeAsync_ptsz ( void* devPtr, cudaStream_t hStream );
cudaError_t __wrap_cudaMallocAsync_ptsz ( void** devPtr, size_t size, cudaStream_t hStream );
cudaError_t __wrap_cudaMemcpyAsync_ptsz ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream );
cudaError_t __wrap_cudaMemsetAsync_ptsz ( void* devPtr, int value, size_t count, cudaStream_t stream );
cudaError_t __wrap_cudaLaunchKernel_ptsz ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream );

cudaError_t __real_cudaFree ( void* devPtr );
cudaError_t __real_cudaFreeHost ( void* ptr );
cudaError_t __real_cudaMalloc ( void** devPtr, size_t size );
cudaError_t __real_cudaMallocHost ( void** ptr, size_t size );
cudaError_t __real_cudaMallocManaged ( void** devPtr, size_t size, unsigned int flags);
cudaError_t __real_cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags );
cudaError_t __real_cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind );
cudaError_t __real_cudaMemset ( void* devPtr, int value, size_t count );
cudaError_t __real_cudaFreeAsync ( void* devPtr, cudaStream_t hStream );
cudaError_t __real_cudaMallocAsync ( void** devPtr, size_t size, cudaStream_t hStream );
cudaError_t __real_cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream );
cudaError_t __real_cudaMemsetAsync ( void* devPtr, int value, size_t count, cudaStream_t stream );
cudaError_t __real_cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream );
cudaError_t __real_cudaMallocAsync_ptsz ( void** devPtr, size_t size, cudaStream_t hStream );
cudaError_t __real_cudaFreeAsync_ptsz ( void* devPtr, cudaStream_t hStream );
cudaError_t __real_cudaMemcpyAsync_ptsz ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream );
cudaError_t __real_cudaMemsetAsync_ptsz ( void* devPtr, int value, size_t count, cudaStream_t stream );
cudaError_t __real_cudaLaunchKernel_ptsz ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream );

};

cudaError_t __wrap_cudaFree ( void* devPtr ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaFree(devPtr);
  printf("TID:%d __real_cudaFree with devPtr:%p, result:%s\n", tid, devPtr, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaFreeHost ( void* ptr ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaFreeHost(ptr);
  printf("TID:%d __real_cudaFreeHost with ptr:%p, result:%s\n", tid, ptr, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMalloc ( void** devPtr, size_t size ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMalloc(devPtr, size);
  printf("TID:%d __real_cudaMalloc with *devPtr:%p size:%d, result:%s\n", tid, *devPtr, size, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMallocHost ( void** devPtr, size_t size ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMallocHost(devPtr, size);
  printf("TID:%d __real_cudaMallocHost with *devPtr:%p size:%d, result:%s\n", tid, *devPtr, size, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMallocManaged ( void** devPtr, size_t size, unsigned int flags)  {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMallocManaged(devPtr, size, flags);
  printf("TID:%d __real_cudaMallocManaged with *devPtr:%p size:%d flags:%d, result:%s\n", tid, *devPtr, size, flags, cudaGetErrorString(result ) );
  return result;
}


cudaError_t __wrap_cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaHostAlloc(pHost, size, flags);
  printf("TID:%d __real_cudaMallocHost with *pHost:%p size:%d flags%d, result:%s\n", tid, *pHost, size, flags, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMemcpy(dst, src, count, kind);
  printf("TID:%d __real_cudaMemcpy with dst:%p src:%p count:%d kind:%d, result:%s\n", tid, dst, src, count, kind, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMemset ( void* devPtr, int value, size_t count ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMemset(devPtr, value, count);
  printf("TID:%d __real_cudaMemset with devPtr:%p value:%d count:%d, result:%s\n", tid, devPtr, value, count, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaFreeAsync ( void* devPtr, cudaStream_t hStream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaFreeAsync(devPtr, hStream);
  printf("TID:%d __real_cudaFreeAsync with devPtr:%p stream:%p, result:%s\n", tid, devPtr, hStream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMemcpyAsync(dst, src, count, kind, stream);
  printf("TID:%d __real_cudaMemcpyAsync with dst:%p src:%p count:%d kind:%d stream:%p, result:%s\n", tid, dst, src, count, kind, stream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMallocAsync ( void** devPtr, size_t size, cudaStream_t hStream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMallocAsync(devPtr, size, hStream);
  printf("TID:%d __real_cudaMallocAsync with *devPtr:%p size:%d hStream:%p, result:%s\n", tid, *devPtr, size, hStream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMemsetAsync ( void* devPtr, int value, size_t count, cudaStream_t stream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMemsetAsync(devPtr, value, count, stream);
  printf("TID:%d __real_cudaMemsetAsync with devPtr:%p value:%d count:%d stream:%p, result:%s\n", tid, devPtr, value, count, stream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  printf("TID:%d __real_cudaLaunchKernel_ptsz with func:%p gridDim(%d,%d,%d) blockDim(%d,%d,%d) sharedMem:%d stream:%p, result:%s\n", 
           tid, func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaFreeAsync_ptsz ( void* devPtr, cudaStream_t hStream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaFreeAsync_ptsz(devPtr, hStream);
  printf("TID:%d __real_cudaFreeAsync_ptsz with devPtr:%p stream:%p, result:%s\n", tid, devPtr, hStream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMemcpyAsync_ptsz ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMemcpyAsync_ptsz(dst, src, count, kind, stream);
  printf("TID:%d __real_cudaMemcpyAsync_ptsz with dst:%p src:%p count:%d kind:%d stream:%p, result:%s\n", tid, dst, src, count, kind, stream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMallocAsync_ptsz ( void** devPtr, size_t size, cudaStream_t hStream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMallocAsync_ptsz(devPtr, size, hStream);
  printf("TID:%d __real_cudaMallocAsync_ptsz with *devPtr:%p size:%d hStream:%p, result:%s\n", tid, *devPtr, size, hStream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaMemsetAsync_ptsz ( void* devPtr, int value, size_t count, cudaStream_t stream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaMemsetAsync_ptsz(devPtr, value, count, stream);
  printf("TID:%d __real_cudaMemsetAsync_ptsz with devPtr:%p value:%d count:%d stream:%p, result:%s\n", tid, devPtr, value, count, stream, cudaGetErrorString(result ) );
  return result;
}

cudaError_t __wrap_cudaLaunchKernel_ptsz ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream ) {
  auto tid = (unsigned int)syscall(SYS_gettid);
  auto result = __real_cudaLaunchKernel_ptsz(func, gridDim, blockDim, args, sharedMem, stream);
  printf("TID:%d __real_cudaLaunchKernel_ptsz with func:%p gridDim(%d,%d,%d) blockDim(%d,%d,%d) sharedMem:%d stream:%p, result:%s\n", 
          tid, func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, cudaGetErrorString(result ) );
  return result;
}
```

1. 关于'`_ptsz`'函数后缀，该行为与"`--default-stream per-thread`"选项有关，在编译*Async*类函数时会被替换成带"`_ptsz`"后缀的函数，请参考`cuda_runtime_api.h`。
2. 根据需要可以添加新的内容，但注意编译选项**也需要**做相应修改。