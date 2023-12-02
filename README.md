# :fire: Alchemy Furanace
本工程基于[OpenMMLab](https://openmmlab.com/codebase)开发，用于个人学习、记录。

[MMEngine 官方文档](https://mmengine.readthedocs.io/zh-cn/latest/)

[MMCV 官方文档](https://mmcv.readthedocs.io/zh-cn/latest/)

[MMDetection 官方文档](https://mmdetection.readthedocs.io/zh-cn/latest/index.html)

[MMDetection3D 官方文档](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html)

******
&nbsp;
&nbsp;
## :bulb: 算法库
&nbsp;
&emsp;

**2D目标检测**

|   Name    | (alchemy)<br>mAP | (alchemy)<br>mF1 | (coco)<br>mAP |                 Experiments                  | Checkpoint |                    Source Code                    |                   Paper                   |
| :-------: | :--------------: | :--------------: | :-----------: | :------------------------------------------: | :--------: | :-----------------------------------------------: | :---------------------------------------: |



&nbsp;
&nbsp;
## :hammer_and_wrench: Install

**步骤0. 创建并激活一个 conda 环境**

~~~bash
conda create --name alchemy python=3.8 -y
conda activate alchemy
~~~

**步骤1. 基于 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装 PyTorch**

在 GPU 平台上：

```bash
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上：

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

**步骤2. 安装MIM**

~~~bash
pip install -U openmim
~~~

**步骤3. 使用mim安装MMEngine**

~~~Bash
mim install mmengine
~~~

**步骤4. 使用mim安装MMCV**

~~~Bash
mim install 'mmcv>=2.0.0rc4'
~~~

**步骤5. 使用mim将 mmdet 作为依赖或第三方 Python 包安装**

~~~Bash
mim install 'mmdet>=3.0.0
~~~

**步骤6. 使用mim将 mmdet3d 作为依赖或第三方 Python 包安装**

~~~Bash
mim install "mmdet3d>=1.1.0rc0"
~~~

**步骤7. 安装相关依赖包**

~~~bash
pip install future tensorboard
~~~

**步骤8. 将alchemy作为Python包以开发模式安装**

~~~bash
python setup.py develop
~~~