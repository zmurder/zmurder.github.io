# 2 定时任务调用串口

有一个需求是在定时任务中调用一个程序usart操作串口。程序usart操作的串口是`/dev/ttyUSB0`。

首先`/dev/ttyUSB0`的权限是777，usart的可执行程序在终端中可以执行并操作`/dev/ttyUSB0`。但是定时任务`cron`却不能执行程序。

定时任务如下

```shell
crontab -l                                    
# Edit this file to introduce tasks to be run by cron.
# 
# Each task to run has to be defined through a single line
# indicating with different fields when the task will be run
# and what command to run for the task
# 
# To define the time you can provide concrete values for
# minute (m), hour (h), day of month (dom), month (mon),
# and day of week (dow) or use '*' in these fields (for 'any').
# 
# Notice that tasks will be started based on the cron's system
# daemon's notion of time and timezones.
# 
# Output of the crontab jobs (including errors) is sent through
# email to the user the crontab file belongs to (unless redirected).
# 
# For example, you can run a backup of all your user accounts
# at 5 a.m every week with:
# 0 5 * * 1 tar -zcf /var/backups/home.tgz /home/
# 
# For more information see the manual pages of crontab(5) and cron(8)
# 
# m h  dom mon dow   command
* * * * * ls -al /dev/ttyUSB* >> /home/hippo/zhaoyd/tt2.txt
* * * * * /home/hippo/1T_disk/zhaoyd/visionperceptiontest/Test/Src/usart/usart >> /home/hippo/zhaoyd/tt2.txt

```

每一分钟第一个定时任务将`/dev/ttyUSB*`的属性输出到`tt2.txt`中，第二个定时任务执行串口程序操作串口输出指令，并将程序打印重定向到`tt2.txt`中。

查看`tt2.txt`结果如下

```shell
~/zhaoyd  cat tt2.txt    
crwxrwxrwx 1 root dialout 188, 0 3月   9 14:27 /dev/ttyUSB0
open /dev/ttyUSB0 
fcntl=0
standard input is not a terminal device
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 0 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 1 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 2 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 3 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 4 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 5 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 6 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 7 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 8 
UART0_Set Error tcgetattr error Bad file descriptor
UART0_Set Error!
Set Port Exactly! 9 
send data failed!
send data failed!

```

可以看到，`/dev/ttyUSB0`操作权限是777，但是程序打开串口显示`Bad file descriptor`

在这里：https://unix.stackexchange.com/questions/325346/how-to-execute-the-command-in-cronjob-to-display-the-output-in-terminal

看到了一句话：

All `cron` jobs are run in non-interactive shells, there is no terminal attachment. Hence the concept of `/dev/tty` or similar is not available in `cron`.

翻译就是所有cron都在非交互式shell中运行，没有终端附件。因此，/dev/tty或类似的概念在cron中不可用。

那么是不是可以将`cron`执行的指令转变为`interactive shell`呢？参考这里[How to launch interactive shell script in bash hourly with cron?](https://askubuntu.com/questions/503028/how-to-launch-interactive-shell-script-in-bash-hourly-with-cron)

参考上面的网址修改我们的定时任务如下

```shell
* * * * * DISPLAY=:0 /usr/bin/gnome-terminal --tab -- /home/hippo/zhaoyd/aa.sh
```

其中aa.sh脚本是为了方便，如下

```
#!/bin/bash

cd /home/hippo/1T_disk/zhaoyd/visionperceptiontest/Test/Src/usart
./usart > /home/hippo/zhaoyd/tt3.txt
```

这样定时任务就可以执行并操作串口了。

下面解释一下`DISPLAY=:0 /usr/bin/gnome-terminal --tab -- `的含义

在Linux/Unix类操作系统上, DISPLAY用来设置将图形显示到何处. 直接登陆图形界面或者登陆命令行界面后使用startx启动图形,  DISPLAY环境变量将自动设置为:0:0, 此时可以打开终端, 输出图形程序的名称(比如xclock)来启动程序, 图形将显示在本地窗口上,  在终端上输入printenv查看当前环境变量, 输出结果中有如下内容:

```shell
DISPLAY=:0.0
```

DISPLAY  环境变量格式如下host:NumA.NumB,host指Xserver所在的主机主机名或者ip地址, 图形将显示在这一机器上,  可以是启动了图形界面的Linux/Unix机器, 也可以是安装了Exceed,  X-Deep/32等Windows平台运行的Xserver的Windows机器. 如果Host为空, 则表示Xserver运行于本机,  并且图形程序(Xclient)使用unix socket方式连接到Xserver,而不是TCP方式. 使用TCP方式连接时,  NumA为连接的端口减去6000的值, 如果NumA为0, 则表示连接到6000端口; 使用unix socket方式连接时则表示连接的unix socket的路径, 如果为0, 则表示连接到/tmp/.X11-unix/X0 . NumB则几乎总是0.





`gnome-terminal`命令用于打开一个新的终端。因此上面的命令整体相当于指定了显示和打开终端。这样就有操作`/dev/tty`的机会了