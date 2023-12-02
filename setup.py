# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.

from setuptools import find_packages, setup

"""
python setup.py命令是用于构建、安装和分发Python包的命令。它需要在Python包的根目录下运行。

常用的一些参数和说明如下:

    build: 构建包,生成构建文件
    install: 安装包
    sdist: 创建源代码分发文件
    bdist: 创建二进制分发文件
    clean: 清除构建文件和缓存文件
    develop: 安装包并支持开发模式,即在安装后仍可编辑源代码
    --user: 将包安装到当前用户目录下

    例如:
        运行以下命令可以构建并安装一个Python包:

        python setup.py build
        python setup.py install

"""

if __name__ == '__main__':
    setup(
        name='alchemy',
        version='1.0',
        author='Ace',
        description='alchemy-furnace',
        packages=find_packages(exclude=('configs', 'tools', 'demo', 'work_dirs', 'experiments'))
        )