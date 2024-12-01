#  1.1 安装[WSL](https://so.csdn.net/so/search?q=WSL&spm=1001.2101.3001.7020)

* 前提条件：我们需要保证你的操作系统版本满足 \*\*Windows 10 版本 2004 及更高版本（内部版本 19041 及更高版本）或 Windows 11 \*\* 才能使用以下命令。

* 启用适用于 Linux 的 Windows 子系统：打开powershell并输入  `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`

* 启用虚拟化:以管理员打开powershell输入下列命令：  `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`
* 可选/推荐】设置WSL默认版本为wsl2：`wsl --set-default-version 2`(WSL2.0相比WSL1.0具备完整的Linux内核、托管VM和完全的系统调用兼容性，所以我们这里使用WSL2.0)
* 直接安装/指定内核版本安装  
* 【直接安装】WSL的安装很简单，可以参考[官方手册](https://learn.microsoft.com/zh-cn/windows/wsl/install#change-the-default-linux-distribution-installed):  可以使用单个命令安装运行 WSL 所需的一切内容。 在管理员模式下打开 PowerShell 或 Windows 命令提示符，方法是右键单击并选择“以管理员身份运行”，输入`wsl --install`命令，然后重启计算机。  【指定内核安装】  当然，可以选择指定内核的方式来安装wsl：

*   若要更改安装的发行版，请输入：`wsl --install -d <Distribution Name>`。 将 `<Distribution Name>`替换为要安装的发行版的名称。
*   若要查看可通过在线商店下载的可用 Linux 发行版列表，请输入：`wsl --list --online`或`wsl -l -o`。
*   若要在初始安装后安装其他 Linux 发行版，还可使用命令：`wsl --install -d <Distribution Name>`。

2 WSL修改默认安装目录到其他盘
=================

显然，此时的wsl默认安装在c盘，随着系统的使用，会占用我们C盘的空间，所以我们将其打包放到其它盘去。

* 查看WSL发行版本  在Windows PowerShell中输入命令:  

  `wsl -l --all -v`  

* 导出分发版为tar文件到d盘  

  `wsl --export Ubuntu-20.04 d:\wsl-ubuntu20.04.tar`(**Ubuntu-20.04修改成你现在的发行版名称**)  

* 注销当前分发版  
  `wsl --unregister Ubuntu-20.04`(**Ubuntu-20.04修改成你现在的发行版名称**)  

* 重新导入并安装WSL在d:\\wsl-ubuntu20.04（**可以修改成你自己想要的目录**）  

  `wsl --import Ubuntu-20.04 d:\wsl-ubuntu20.04 d:\wsl-ubuntu20.04.tar --version 2`  

* 设置默认登陆用户为安装时用户名  

  `ubuntu2004 config --default-user Username`  

* 删除tar文件(可选)  
* `del d:\wsl-ubuntu20.04.tar`

经过以上操作后，就将WSL的默认安装目录迁移到D:\\wsl-ubuntu20.04目录(可以自己修改自己想要的目录)下了。此目录即为WSL的根文件系统。

本文转自 <https://blog.csdn.net/farer_yyh/article/details/133934904>，如有侵权，请联系删除。