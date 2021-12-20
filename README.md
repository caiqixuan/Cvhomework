# Cvhomework
请下载整个文件的压缩包

#### 安装库

~~~shell
torch
torchvision
argparse
time
opencv-python
tensorboardX 
os
logging
numpy
sys
random
shutil
typing
~~~

#### 下载WFLW数据集

1. WFLW 训练和测试图片 [[Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)] [[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)]
2. WFLW  [Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz)
3. 将这两个文件解压并放在路径 `./data/WFLW/`
4. 把 `Mirror98.txt` 移到 `WFLW/WFLW_annotations`

~~~shell
$ cd data # cd 到Cvhomework/data 路径下
$ python3 SetPreparation.py
~~~

#### 训练和测试

训练:

~~~shell
$ cd ./Cvhomework
$ python3 train.py
~~~

测试:

~~~shell
$ cd ./Cvhomework
$ python3 test.py
~~~

