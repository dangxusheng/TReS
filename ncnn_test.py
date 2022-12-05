#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: ncnn_test.py
@time: 2022/11/16 上午10:51
@desc:
"""

"""
python ncnn_test.py
"""

import cv2
import math
import numpy as np
from ncnn_basenet import NCNNBaseNet
import time

# overwrite torch function
class torch:
    int = np.int32
    long = np.int64
    @classmethod
    def cat(cls, datas, dim=-1):
        return np.concatenate(datas, axis=dim)

    @classmethod
    def clamp(cls, data, min, max):
        return np.clip(data, min, max)

    @classmethod
    def mean(cls, data, dim=-1):
        return np.mean(data, axis=dim)

    @classmethod
    def sum(cls, data, dim):
        return np.sum(data, axis=dim)

    @classmethod
    def argmax(cls, data, dim=-1):
        return np.argmax(data, axis=dim)

    @classmethod
    def unsqueeze(cls, data, dim=-1):
        return np.expand_dims(data, axis=dim)

    @classmethod
    def where(cls, data):
        return np.where(data)

    @classmethod
    def triu(cls, data, diagonal=1):
        return np.triu(data, k=diagonal)

    @classmethod
    def norm(cls, data, dim=1):
        return np.linalg.norm(data, axis=dim)

    @classmethod
    def acos(cls, data):
        return np.arccos(data)

    @classmethod
    def sin(cls, data):
        return np.sin(data)

    @classmethod
    def arange(cls, start, end, step=1):
        return np.arange(start, end, step)

    @classmethod
    def meshgrid(cls, *xi, **kwargs):
        return np.meshgrid(*xi, kwargs)

    @classmethod
    def sqrt(cls, data):
        return np.sqrt(data)

    @classmethod
    def unique(cls, data, dim=-1):
        return np.unique(data, axis=dim)


class NCNN_IQANet(NCNNBaseNet):
    CLASSES = ('junction', '__backgound__',)

    MODEL_ROOT = 'Save_Models_TReS/tid2013_1_2021/sv'
    PARAM_PATH = f'{MODEL_ROOT}/bestmodel_1_2021.onnx.opt.param'
    BIN_PATH = f'{MODEL_ROOT}/bestmodel_1_2021.onnx.opt.bin'

    INPUT_W = 224
    INPUT_H = 224
    INPUT_C = 3
    MEAN = [0., ] * INPUT_C
    STD = [1 / 255., ] * INPUT_C

    OUTPUT_NODES = [
        '944',
    ]


    def detect(self, img, thres=0.7):
        src_size_hw = img.shape[:2]
        mat_in = self.preprocess(img)
        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)
        ex.input(self.input_names[0], mat_in)

        s = time.time()
        outs = []
        for node in self.OUTPUT_NODES:
            assert node in self.output_names, f'{node} not in {self.output_names}'
            ret, out = ex.extract(node)  # [n,k,k]
            out = np.asarray(out)
            out = out[None, ...]
            # print(out.shape)
            outs.append(out)
        print(f'cnn forward() elasped : {time.time() - s} s', )
        mat_in.release()
        return outs


if __name__ == "__main__":
    print('hello')
    # x = cv2.imread('./1514.png')
    x = np.random.randint(0, 255, [128, 128, 3], dtype=np.uint8)
    m = NCNN_IQANet()

    s = time.time()
    iqa = m.detect(x)
    print(f'detect() elasped : {time.time() - s} s', )

