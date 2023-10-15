# 1 protobuf概述

本教程基本上时官方教程的翻译

## 1.1 是什么

Google Protocol Buffer(简称 Protobuf)是一种轻便高效的结构化数据存储格式，平台无关、语言无关、可扩展，可用于**通讯协议**和**数据存储**等领域。

## 1.2 为什么用

* 平台无关，语言无关，可扩展；
* 提供了友好的动态库，使用简单；
* 解析速度快，比对应的XML快约20-100倍；
* 序列化数据非常简洁、紧凑，与XML相比，其序列化之后的数据量约为1/3到1/10。

## 1.3 跨语言兼容性

可以使用一个jave程序序列化数据发送给python程序解析

## 1.4 向前向后兼容新

按照[规则](https://developers.google.cn/protocol-buffers/docs/proto3#updating)进行升级proto后（规则在后面proto3语言指南中由描述），旧的代码可以读取新的消息，忽略新添加的字段。对于旧代码，被删除的字段将具有其默认值，而被删除的重复字段将为空。

新代码也可以阅读旧消息。旧消息中不会出现新的字段;在这些情况下，protobuf会提供合理的默认值。

设置可选条件和字段类型后，您会分配字段编号。**字段编号不能重新估算或重复使用。如果删除字段，则应保留其字段编号，以防止某人意外地重用该数字**。

**实际测试proto2和proto3也可以兼容，例如一个主机使用proto2发送另一台主机使用proto3接收，相同的proto文件，分别使用proto2和proto3编译为cc和h，通信成功。**



## 1.5 编译安装

源码下载地址： https://github.com/google/protobuf 
安装依赖的库： autoconf automake libtool curl make g++ unzip 

```shell
$ ./autogen.sh
$ ./configure
$ make
$ make check
$ sudo make install
```



## 1.6 额外的数据类型支持

- [`Duration`](https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/duration.proto)  is a signed, fixed-length span of time, such as 42s. 
- [`Timestamp`](https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/timestamp.proto)  is a point in time independent of any time zone or calendar, such as 2017-01-15T01:30:15.01Z.
- [`Interval`](https://github.com/googleapis/googleapis/blob/master/google/type/interval.proto)  is a time interval independent of time zone or calendar, such as 2017-01-15T01:30:15.01Z - 2017-01-16T02:30:15.01Z. 
- [`Date`](https://github.com/googleapis/googleapis/blob/master/google/type/date.proto)  is a whole calendar date, such as 2025-09-19.
- [`DayOfWeek`](https://github.com/googleapis/googleapis/blob/master/google/type/dayofweek.proto)  is a day of the week, such as Monday.
- [`TimeOfDay`](https://github.com/googleapis/googleapis/blob/master/google/type/timeofday.proto)  is a time of day, such as 10:42:23.
- [`LatLng`](https://github.com/googleapis/googleapis/blob/master/google/type/latlng.proto)  is a latitude/longitude pair, such as 37.386051 latitude and -122.083855 longitude.
- [`Money`](https://github.com/googleapis/googleapis/blob/master/google/type/money.proto)  is an amount of money with its currency type, such as 42 USD.
- [`PostalAddress`](https://github.com/googleapis/googleapis/blob/master/google/type/postal_address.proto)  is a postal address, such as 1600 Amphitheatre Parkway Mountain View, CA 94043 USA.
- [`Color`](https://github.com/googleapis/googleapis/blob/master/google/type/color.proto)  is a color in the RGBA color space.
- [`Month`](https://github.com/googleapis/googleapis/blob/master/google/type/month.proto)  is a month of the year, such as April.

# 