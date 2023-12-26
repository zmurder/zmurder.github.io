# 1 简介

很多时候，可能需要替换已有库函数，或者对其库函数进行修改。为了避免对那些静态链接库或者动态链接库文件大动干戈，我们可以使用ld提供的–wrap选项。

参考 https://www.man7.org/linux/man-pages/man1/ld.1.html 中`--wrap=symbol`

# 2 示例

例如，想把所有的malloc函数都作修改，以便让malloc出的内存都是32字节对齐的。我们可以给ld传选项“­­wrap=malloc”， 告诉ld，我们将替换名称为malloc的函数。接着再如下定义一个新的malloc函数:

```C++
void * __wrap_malloc( size_t size) {
  void* result;
  result=memalign(32, size);
  return result;
}
```

可以看到，程序中有个类似库函数名称的__wrap_malloc。ld在遇到__wrap选项后，会使用__wrap_malloc函数名替换所有对malloc的调用。并以此实现替换的作用。

那麽，如果还向调用原函数怎么办呢？可以使用__real_malloc，这个函数名称就对应了原来的malloc。 每次调用malloc时都打印生成的指针地址。

```c++
void * __wrap_malloc( size_t size) {
  void* result;
  result= __real_malloc( size);
  printf("Malloc: allocate %d byte mem, start at %p", size, result);

  return result;
}
```

附完整程序：

```c++
#include <stdio.h>
#include <stdlib.h>

void * __wrap_malloc( size_t size) {
  void* result;
  //result= __real_malloc( size);   result = memalign(32, size);
  printf("Malloc: allocate %d byte mem, start at %p", size, result);

  return result;
}

int main ()
{
  int i,n;
  char * buffer;

  printf ("How long do you want the string? ");
  //scanf ("%d", &i);   i = 200;

  buffer = (char*) malloc (i+1);
  if (buffer==NULL) exit (1);

  for (n=0; n<i; n++)
    buffer[n]=rand()%26+'a';
  buffer[i]='\0';

  printf ("Random string: %s\n",buffer);
  free (buffer);

  return 0;
}
```

编译选项

```shell
$gcc test_malloc.c -Wl,--wrap=malloc
```

