# 1 常用的gcc编译选项

* 没有编译选项 `gcc  helloworld.c`结果会在与helloworld.c相同的目录下产生一个a.out的可执行文件。
* `-Wall`选项打开所有最常用到的编译警告，强烈建议打开，可以捕捉到许多在C编程中最常发生的错误。比如warning: control reaches end of non-void function （警告：控制流到达返回值非void的函数结尾）。

* `-o`  选项来为可执行文件指定一个不同的输出文件。`gcc -o helloworld helloworld.c `输出的可执行文件的名为helloworld

* `-c` 只编译，不汇编连接。`gcc -c helloworld.c`产生一个叫helloworld.o的目标文件

* `-l` `-lNAME`试图链接标准库目录下的文件名为`libNAME.a`中的对象文件。在大型程序中通常会用到很多`-l`选项，来链接象数学库，图形库和网络库。使用选项`-lNAME`的情况下，静态库`libNAME`可以用于链接，但编译器首先会检查具有相同名字和`.so`为扩展名的共享库。默认情况下，载入器仅在一些预定义的系统目录中查找共享库，比如`/usr/local/lib`和`/usr/lib`。如果库不在这些目录中，那它必须被添加到载入路径（load path）中去。设置载入路径的最简单方法是通过环境变量LD_LIBRARY_PATH。

* `-static`选项可以迫使gcc静态链接，避免使用共享库。

* `-I`用于把新目录添加到include路径上。例如，`-I/opt/gdbm-1.8.3/include`

* `-L`用于把新目录添加到库搜索路径上。例如，`-L/opt/gdbm-1.8.3/lib`。

* `-ansi`禁止那些与ANSI/ISO标准冲突的GNU扩展特性。

* `-std`选项来控制GCC编译时采用的某个C语言标准。

* `-DNAME`选项在命令行上定义预处理宏NAME，默认情况下，其值为1。`-D`命令行选项可以用来定义有值的宏，形式是`-DNAME=VALUE`，例如-DNUM="2+2"，预处理器将把NUM替换成2+2。当宏是某个表达式的一部分时，用圆括号把宏括起来是个好主意，比如10 \* (NUM)*
* `-g`调试选项来在对象文件和可执行文件中存储另外的调试信息。
* `-OLEVEL`用来选择哪一种优化级别，这里LEVEL是从1到3的数字。`-O0`或没有`-O`选项（默认），在该优化级别，GCC不会实施任何优化。

# 附录

官方：https://gcc.gnu.org/onlinedocs/gcc-13.2.0/gcc/Option-Index.html