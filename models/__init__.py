"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
# generators
from .generator import Generator

# discriminator builder
from .discriminator import disc_builder

def generator_dispatch():
    return Generator
