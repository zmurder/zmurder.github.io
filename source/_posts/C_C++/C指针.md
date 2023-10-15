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

**解释一下为什么上面程序不能运行：首先 `str`是一个指针，指向一个地址，假如`str`指向的地址是`0x0001` ，那么`p`指向的地址也是`0x0001`。但是`str`和`p`自身的地址是不一致的，因为函数是值传递，拷贝了一个副本。因此采用上面的内存分配方式，当分配好新的内存之后，`p`指向的已经是新内存的首地址，而`str`依然是指向`0x0001`。**

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
//testPoint3.c
#include <stdio.h>
#include<stdlib.h>
#include <string.h>
 
void P1Malloc(char* p,int num)
{
	p=malloc(num);
	printf("P1Malloc,current malloc address p:%p\n",p);
}
 
void P2Malloc(char** p,int num)
{
	*p=malloc(num);
	printf("P2Malloc,current malloc address *p:%p\n",*p);
}

char* P3Malloc(char* p,int num)
{
	p=malloc(num);
	printf("P3Malloc,current malloc address p:%p\n",p);
	return p;
}

int main()
{
	int Num=10;
	char* a=NULL;
	
	printf("initial pointer a:%p\n",a);
	
	P1Malloc(a,Num);
	// strcpy(a,"hello");//指针a 并没有变化，没有申请空间，调用会段错误
	printf("after using *,ponter a:%p\n",a);
	
	P2Malloc(&a,Num);
	printf("after using **,ponter a:%p\n",a);
	strcpy(a,"hello");
	free(a);
	
	a = P3Malloc(a,Num);
	printf("after P3Malloc,ponter a:%p\n",a);
	strcpy(a,"hello");
	free(a);
	
	return 0;
}
```

编译运行如下

```shell
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ gcc -o testPoint3 testPoint3.c 
zmurder@zmurder:~/WorkSpace/zyd/test/C C++$ ./testPoint3
initial pointer a:(nil)
P1Malloc,current malloc address p:0x555b5a5c06b0
after using *,ponter a:(nil)
P2Malloc,current malloc address *p:0x555b5a5c06d0
after using **,ponter a:0x555b5a5c06d0
P3Malloc,current malloc address p:0x555b5a5c06d0
after P3Malloc,ponter a:0x555b5a5c06d0
```

分析：

* 原始的a的是nil（注意`a`是一个指针变量，也就是一个变量，存的是地址）
* P1Malloc函数中`p`的地址是0x555b5a5c06b0，并没有改变`a`。
* P2Malloc函数中`*p`的地址是0x555b5a5c06d0，改变了`a`，也是0x555b5a5c06d0
* P3Malloc函数中`p`的地址是0x555b5a5c06d0，函数返回后赋值给`a`，因次`a`改变为0x555b5a5c06d0





# 2 指针数组与数组指针

* 指针数组：指针数组可以说成是”指针的数组”，首先这个变量是一个数组，其次，”指针”修饰这个数组，意思是说这个数组的所有元素都是指针类型，在32位系统中，指针占四个字节。
* 数组指针：数组指针可以说成是”数组的指针”，首先这个变量是一个指针，其次，”数组”修饰这个指针，意思是说这个指针存放着一个数组的首地址，或者说这个指针指向一个数组的首地址。

## 2.1指针数组

首先先定义一个指针数组，既然是数组，名字就叫arr。（注意优先级：()>[]> *）

```C
char *arr[4] = {"hello", "world", "shannxi", "xian"};
//可以写成char *(arr[4])
//arr就是我定义的一个指针数组，它有四个元素，每个元素是一个char *类型的指针，这些指针存放着其对应字符串的首地址。
```

这个指针数组是16个字节，因为它是一个指针数组。

相当与定义`char *p1 = “hello”，char *p1 = “world”，char *p3 = “shannxi”， char  *p4 = “xian”`，这是四个指针，每个指针存放一个字符串首地址，然后用`arr[4]`这个数组分别存放这四个指针，就形成了指针数组。

## 2.2数组指针

首先来定义一个数组指针，既然是指针，名字就叫pa

```C
char (*pa)[4];
```

既然pa是一个指针，存放一个数组的地址，那么在我们定义一个数组时，数组名称就是这个数组的首地址，那么这二者有什么区别和联系呢？

```C
char a[4];
```

* a是数组首元素首地址，a是char 类型，a+1，地址加1
* pa存放的却是数组首地址，而pa是char[4]类型的，pa+1，地址加4

## 2.3 指针数组使用

指针数组常用在`函数传参`

```C
int main(int argc, char *argv[])
```

```C
void fun(char **pp);//子函数中的形参
fun(char *p[]);//主函数中的实参
```

下面是一个例子：

```C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void change(char **ptr) {
	ptr[2] = "student"; //正确
	//*ptr[2] = "student";//错误
	//*(ptr+2) = "student";//正确 第2个指针指向的内容变化，相当于原变量的buf[2]
}
 
void main() {
	char *buf[4] = { "hello","world","welcome","profect" };//指针数组，数组里面是4个指针
	char **ptr = buf;
	int i = 0;
 
	change(buf);
 
	for (i = 0; i < 4; i++) {
		printf("%s\n", *(ptr +i)); //使用二级指针来遍历字符指针数组；
	}
	
}
```

运行结果如下

```bash
$ ./testPointArray 
hello
world
student
profect
```

## 2.4 数组指针的使用

```C
#include<stdio.h>
#include<stdlib.h>
void main()
{
    int a[3][4]={{1,2,3,4},{11,12,13,14},{21,22,23,24}};
    int (*p)[4]; //该语句是定义一个数组指针，指针步长为4个int即16位。
    p=a;
    int i=0;
    while(i<3)
    {
        //printf("%d\t",(*p)[i]);
        //数组指针，指向的是一个数组整体，相当于指针也带了下标，当执行i++操作时，下标+1，得到该数组的下一个元素，
        //在该例中，指针没有位移，所以依次输出为1 2 3

        printf("%d\t",(*p++)[0]);
        //整型数组类型的指针，指向的是一个数组整体，当执行*p++操作时，指针位移该数组长度的位数
        //在该例中，即指针位移4个int的长度，所以输出是1 11 21
        i++;
    }
}
```



附录：

[指针的指针作用（申请空间）之一](https://blog.csdn.net/hanchaoman/article/details/4137340)

[【CUDA】分配内存使用void**](https://blog.csdn.net/TwT520Ly/article/details/81100301)

[CUDA 中的 cudaMalloc使用二重指针（void**)的一些理解](https://blog.csdn.net/lingyunxianhe/article/details/92001270)

[数组指针的用法，用处。](https://www.cnblogs.com/anwcq/p/p_shuzuzhizhen.html)

[6.指针数组做函数参数](https://blog.csdn.net/wangfan110/article/details/118336751)