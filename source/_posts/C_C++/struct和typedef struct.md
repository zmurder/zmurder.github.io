# 1 typedef

`typedef`为C语言的关键字，作用是为一种数据类型定义一个新名字。这里的数据类型包括内部数据类型（`int,char`等）和自定义的数据类型（`struct`等）。在编程中使用`typedef`目的一般有两个，一个是给变量一个易记且意义明确的新名字，另一个是简化一些比较复杂的类型声明。例如：

```C
typedef long byte_4;//给已知数据类型long起个新名字，叫byte_4
```

# 2 typedef与struct结合使用

```C
typedef struct tagMyStruct
{ 
　int iNum;
　long lLength;
} MyStruct;
```

这段代码实际上完成两个操作：

1. 定义一个新的结构体类型`tagMyStruct`

   ```C++
   struct tagMyStruct
   { 
   　int iNum; 
   　long lLength; 
   };
   struct tagMyStruct varName;//C语言中 varName是变量 struct tagMyStruct是变量类型
   tagMyStruct varName//C++中 varName是变量 tagMyStruct是变量类型
   ```

2. `typedef`为这个新的结构起了一个名字，叫`MyStruct`。

   ```C 
   typedef struct tagMyStruct MyStruct;
   ```

   因此，`MyStruct`实际上相当于`struct tagMyStruct`，可以使用`MyStruct varName`来定义变量。

# 3 C和C++的区别

在C中定义一个结构体类型要用`typedef`:

```C
//C语言中 定义结构体
typedef struct Student
{
	int a;
}Stu;
//声明结构体变量
struct Student stu1;
Stu stu2;//如果没有typedef就必须用struct Student stu1;来声明。Stu实际上就是struct Student的别名
```

```C
//另外这里也可以不写Student（于是也不能struct Student stu1;了，必须是Stu stu1;）
typedef struct
{
	int a;
}Stu;
//声明结构体变量
Stu stu1;
```

c++中

```C
//但在c++里很简单，直接定义结构体类型Student
struct Student
{
	int a;
};　　　
//声明变量时直接
Student stu2；
```

```
struct Student  
{  
	int a;  
}stu1;//stu1是一个变量
```

```C
//在c++中如果用typedef的话，又会造成区别
typedef struct Student2  
{  
　　int a;  
}stu2;//stu2是一个结构体类型=struct Student
//声明结构体变量
struct Student2 s1;
stu2 s2;
```

# 4 总结

```C
typedef struct tagMyStruct
{
    int iNum;
	long lLength;
} MyStruct;
```

* 在C中，这个申明后申请结构变量的方法有两种：

  ```C 
  struct tagMyStruct variableName;
  MyStruct variableName;
  ```

* 在c++中可以有

  ```C
  struct tagMyStruct variableName;
  MyStruct variableName;
  tagMyStruct variableName;
  ```

  

