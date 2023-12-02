# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright (c) 2023 by Ace, All Rights Reserved.  

import datetime


def heading_line(log):
    print('{:-^60}'.format(f' {log} '))


def format_datetime(seconds):
    return str(datetime.timedelta(seconds=seconds))
