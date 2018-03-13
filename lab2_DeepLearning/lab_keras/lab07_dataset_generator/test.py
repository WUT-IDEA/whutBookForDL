# -*- coding: utf-8 -*-
# !/usr/bin/env python
'''
@author: Yang
@time: 18-2-24 下午2:08
'''

from __future__ import print_function

import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np
import os


def fake_data_generator(num=100):
    # define make direction func
    def mkdir(name=None):
        if os.path.exists(name):
            pass
        else:
            os.mkdir(name)

    # generate images
    mkdir(name='images')
    for i in xrange(num):
        new_array = np.zeros(shape=(10, 10), dtype=np.float32) * num
        np.save(file='images/%s.npy' % i, arr=new_array)
    # generate text
    mkdir(name='text')
    for i in xrange(num):
        new_text = str(i)
        with open('text/%s.txt' % i, mode='wb') as text_buffer:
            text_buffer.write(new_text)


fake_data_generator()


class Generator():
    def __init__(self, img_dir, text_dir, batch_size=10):
        self.batch_index = 0
        self.img_dir = img_dir
        self.text_dir = text_dir
        self.batch_size = batch_size
        assert len(os.listdir(self.text_dir)) == len(os.listdir(self.text_dir))
        self.NUM = len(os.listdir(self.text_dir))

    def next(self):
        batch_img = ['%s/%s.npy' % (self.img_dir, i % self.NUM) for i in \
                     xrange(self.batch_index, self.batch_index + self.batch_size)]
        batch_text = ['%s/%s.txt' % (self.text_dir, i % self.NUM) for i in \
                      xrange(self.batch_index, self.batch_index + self.batch_size)]
        self.batch_index = (self.batch_index + self.batch_size) % self.NUM
        return (batch_img, batch_text)

    def __len__(self):
        return self.NUM // self.batch_size


gen = Generator(img_dir='images', text_dir='text')
for _ in xrange(len(gen)):
    print(next(gen))
