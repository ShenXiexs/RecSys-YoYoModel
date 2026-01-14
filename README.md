### 如何实现自己的模型训练
#### step1
在models目录下新建自己的模型py，里面定义好model_fn
#### step2
在config目录下创建自己的模型名的目录，例如“TC1”，然后配置train_config.py，可参考dome/train_config.py;

还需要配置fg.json,model.config(上线时配),schema.conf,slot.conf,select_feature.conf等

| 文件名                 | 作用                |
|---------------------|-------------------|
| train_config.py     | 上线时需要配置路径信息       |
| fg.json             | 跟构建样本的fg.json对其   |
| schema.conf         | 样本数据表schema       |
| slot.conf           | 从schema中挑选出的特征下载  |
| select_feature.conf | dataset加载slot中的特征 |
#### step3-执行
```text
例如:
运行config下TC1的配置训练TC1模型;
训练书周期：[20250101,20250101];
是否批量训练：1;数字表示批量周期,1表示用1天数据每日批量训练，这个数字只有在输入开始和结束日期相同的情况下才会计算周期，否则只表示是否批量训练;例：输入20250101 20250101 1只用1号数据批量训练，20250102 20250102 2用1-2号数据批量训练，输入20250101 20250105 2用1-5号数据批量训练；
cd bin & bash run.sh TC1 20250101 20250101 1
```

### 3 分钟了解如何进入开发

欢迎使用云效代码管理 Codeup，通过阅读以下内容，你可以快速熟悉 Codeup ，并立即开始今天的工作。

### 提交**文件**

Codeup 支持两种方式进行代码提交：网页端提交，以及本地 Git 客户端提交。

* 如需体验本地命令行操作，请先安装 Git 工具，安装方法参见[安装Git](https://help.aliyun.com/document_detail/153800.html)。

* 如需体验 SSH 方式克隆和提交代码，请先在平台账号内配置 SSH 公钥，配置方法参见[配置 SSH 密钥](https://help.aliyun.com/document_detail/153709.html)。

* 如需体验 HTTP 方式克隆和提交代码，请先在平台账号内配置克隆账密，配置方法参见[配置 HTTPS 克隆账号密码](https://help.aliyun.com/document_detail/153710.html)。

现在，你可以在 Codeup 中提交代码文件了，跟着文档「[__提交第一行代码__](https://help.aliyun.com/document_detail/153707.html?spm=a2c4g.153710.0.0.3c213774PFSMIV#6a5dbb1063ai5)」一起操作试试看吧。

<img src="https://img.alicdn.com/imgextra/i3/O1CN013zHrNR1oXgGu8ccvY_!!6000000005235-0-tps-2866-1268.jpg" width="100%" />


### 进行代码检测

开发过程中，为了更好的维护你的代码质量，你可以开启 Codeup 内置开箱即用的「[代码检测服务](https://help.aliyun.com/document_detail/434321.html)」，开启后提交或合并请求的变更将自动触发检测，识别代码编写规范和安全漏洞问题，并及时提供结果报表和修复建议。

<img src="https://img.alicdn.com/imgextra/i2/O1CN01BRzI1I1IO0CR2i4Aw_!!6000000000882-0-tps-2862-1362.jpg" width="100%" />

### 开展代码评审

功能开发完毕后，通常你需要发起「[代码评审并执行合并](https://help.aliyun.com/document_detail/153872.html)」，Codeup 支持多人协作的代码评审服务，你可以通过「[保护分支设置合并规则](https://help.aliyun.com/document_detail/153873.html?spm=a2c4g.203108.0.0.430765d1l9tTRR#p-4on-aep-l5q)」策略及「[__合并请求设置__](https://help.aliyun.com/document_detail/153874.html?spm=a2c4g.153871.0.0.3d38686cJpcdJI)」对合并过程进行流程化管控，同时提供在线代码评审及冲突解决能力，让评审过程更加流畅。

<img src="https://img.alicdn.com/imgextra/i1/O1CN01MaBDFH1WWcGnQqMHy_!!6000000002796-0-tps-2592-1336.jpg" width="100%" />

### 成员协作

是时候邀请成员一起编写卓越的代码工程了，请点击左下角「成员」邀请你的小伙伴开始协作吧！

### 更多

Git 使用教学、高级功能指引等更多说明，参见[Codeup帮助文档](https://help.aliyun.com/document_detail/153402.html)。