# 2 protobuf proto3语言指南

## 2.1 导入定义（import）

Import 可选项用于包含其它 proto 文件中定义的 message或 enum等。标准格式如 下

```C
import "info.proto";
```

info.proto文件内容如下：

```C
syntax = "proto3";//指定版本信息，不指定会报错

package infopack; //package声明符

message info //message为关键字，作用为定义一种消息类型
{
    string addr = 1;    //地址
    string group = 2;   //分组
}
```

addressbook.proto文件内容如下，addressbook.proto文件需要导入info.proto文件的内容：

```C
syntax = "proto3";//指定版本信息，不指定会报错

import "info.proto"; //导入定义

package tutorial; //package声明符

message Person //message为关键字，作用为定义一种消息类型
{
    string name = 1;    //姓名
    int32 id = 2;       //id
    string email = 3; //邮件

    enum PhoneType //枚举消息类型
    {
        MOBILE = 0; //proto3版本中，首成员必须为0，成员不应有相同的值
        HOME = 1;
        WORK = 2;
    }

    message PhoneNumber
    {
        string number = 1;
        PhoneType type = 2;
    }

    repeated PhoneNumber phones = 4; //phones为数组

    //info定义在"info.proto"
    //类型格式：包名.信息名
    infopack.info tmp = 5;
}

message AddressBook
{
    repeated Person people = 1;
}
```

编译proto文件

````C
protoc -I=./ --cpp_out=./ *.proto
````

测试程序

```C
#include "addressbook.pb.h"
#include <iostream>
#include <fstream>
using namespace std;

void set_addressbook()
{
    tutorial::AddressBook obj;

    tutorial::Person *p1 = obj.add_people(); //新增加一个Person
    p1->set_name("mike");
    p1->set_id(1);
    p1->set_email("mike@qq.com");

    tutorial::Person::PhoneNumber *phone1 = p1->add_phones(); //增加一个phone
    phone1->set_number("110");
    phone1->set_type(tutorial::Person::MOBILE);

    tutorial::Person::PhoneNumber *phone2 = p1->add_phones(); //增加一个phone
    phone2->set_number("120");
    phone2->set_type(tutorial::Person::HOME);

    //info addr和group的使用
    infopack::info *p_info = p1->mutable_tmp(); //取出info的对象指针
    p_info->set_addr("China");  //地址
    p_info->set_group("A");     //组

    fstream output("pb.xxx", ios::out | ios::trunc | ios::binary);

    bool flag = obj.SerializeToOstream(&output);//序列化
    if (!flag)
    {
        cerr << "Failed to write file." << endl;
        return;
    }

    output.close();//关闭文件
}

void get_addressbook()
{
    tutorial::AddressBook obj;
    fstream input("./pb.xxx", ios::in | ios::binary);
    obj.ParseFromIstream(&input);  //反序列化
    input.close(); //关闭文件

    for (int i = 0; i < obj.people_size(); i++)
    {
        const tutorial::Person& person = obj.people(i);//取第i个people
        cout << "第" << i + 1 << "个信息\n";
        cout << "name = " << person.name() << endl;
        cout << "id = " << person.id() << endl;
        cout << "email = " << person.email() << endl;

        for (int j = 0; j < person.phones_size(); j++)
        {
            const tutorial::Person::PhoneNumber& phone_number = person.phones(j);

            switch (phone_number.type())
            {
            case tutorial::Person::MOBILE:
                cout << "  Mobile phone #: ";
                break;
            case tutorial::Person::HOME:
                cout << "  Home phone #: ";
                break;
            case tutorial::Person::WORK:
                cout << "  Work phone #: ";
                break;
            }

            cout << phone_number.number() << endl;
        }

        //info addr和group的使用
        infopack::info info = person.tmp(); //取出info的对象
        cout << "addr = " << info.addr() << endl;
        cout << "group = " << info.group() << endl;

        cout << endl;
    }
}

int main()
{
    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    set_addressbook(); //序列化
    get_addressbook(); //反序列化

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
```



# 