# 1 二重指针

**二重指针**的C语言表示：`int **p`,它表示**指针的指针**。具体什么用途呢？先看一个例子

以下是经典程序（载自林锐的从c/c++高质量编程）

```C
//testPoint.c 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void GetMemory(char *p,int num)
{
	p=(char*)malloc(sizeof(char)*num); //p是形参指向的地址
}
void main()
{
	char *str=NULL;
	GetMemory(str,100); //str是实参指向的地址，不能通过调用函数来申请内存
	strcpy(str,"hello");
	
	printf("str=%s\n",str);
}
```

结构是编译能通过，却不能运行，如下

```shell
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ gcc -o testPoint testPoint.c 
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ ls
testPoint  testPoint.c
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ ./testPoint 
Segmentation fault

```

解释一下为什么上面程序不能运行：首先 `str`是一个指针，指向一个地址，假如`str`指向的地址是`0x0001` ，那么`p`指向的地址也是`0x0001`。但是`str`和`p`自身的地址是不一致的，因为函数是值传递，拷贝了一个副本。因此采用上面的内存分配方式，当分配好新的内存之后，`p`指向的已经是新内存的首地址，而`str`依然是指向`0x0001`。

对上面的程序进行修改如下

```C
//testPoint2.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void GetMemory(char **p,int num)
{
	*p=(char*)malloc(sizeof(char)*num); //此时*p就变成了是形参本身的地址
}
void main()
{
	char *str=NULL;
	GetMemory(&str,100); //&str是实参的地址，所以实参和形参之间就可以直接调用
	strcpy(str,"hello");
	printf("str=%s\n",str);
	free(str); 
}
```

相当于p是指向`str`的地址的指针，因此`*p = str`，给`*p`分配内存，相当于让`str`指向新内存的首地址。

编译运行如下

```shell
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ gcc -o testPoint2 testPoint2.c 
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ ./testPoint2 
str=hello
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ 
```

一般指针的指针用作参数，**大多用在需要函数改变指针(重新引用变量)而又不能通过返回值传递(例如返回值用于传递其他结果)时**。

例如：

* `void * malloc(size_t size)`函数返回的就是一个指针。调用后有返回值，直接修改了原变量（内容是地址）。` char * p ; p=malloc(size);`
* 自己写一个函数 `void GetMemory(void ** p ,size_t size)`。返回值是空的，但是想修改一个变量的地址（注意修改的是地址）。只能将变量的地址的地址传进去`char ** p ; GetMemory((void**)(&p),size);`。

再看一个例子

```C
//testPoint3.c
#include <stdio.h>
#include<stdlib.h>
 
void P1Malloc(int* p)
{
	p=(int*)malloc(10);
	printf("P1Malloc,current malloc address:%p\n",p);
}
 
void P2Malloc(void** p)
{
	*p=malloc(10);
	printf("P2Malloc,current malloc address:%p\n",*p);
}
 
int main()
{
	int Num=10;
	int* a=&Num;
	printf("initial pointer a:%p\n",a);
	P1Malloc(a);
	printf("after using *,ponter a:%p\n",a);
	P2Malloc((void**)&a);
	printf("after using **,ponter a:%p\n",a);
 
	return 0;
}
```

编译运行如下

```shell
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ gcc -o testPoint3 testPoint3.c 
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ ./testPoint3
initial pointer a=0x7ffc6d8fc4cc

P1Malloc,current malloc address=0x556283dcd6b0
after using *,ponter a=0x7ffc6d8fc4cc
P2Malloc,current malloc address=0x556283dcd6d0
after using **,ponter a=0x556283dcd6d0
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$
```

分析：

* 原始的a的地址是0x7ffc6d8fc4cc
* P1Malloc函数中`p`的地址是0x556283dcd6b0，并没有改变`a`的地址。需要注意的是函数
* P2Malloc函数中`*p`的地址是0x556283dcd6d0，改变了`a`的地址

附录：

[指针的指针作用（申请空间）之一](https://blog.csdn.net/hanchaoman/article/details/4137340)

[【CUDA】分配内存使用void**](https://blog.csdn.net/TwT520Ly/article/details/81100301)

[CUDA 中的 cudaMalloc使用二重指针（void**)的一些理解](https://blog.csdn.net/lingyunxianhe/article/details/92001270)