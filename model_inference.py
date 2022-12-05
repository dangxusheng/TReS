#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site:
@software: PyCharm
@file: model_inference.py
@time: 2022/11/30 下午1:04
@desc: 使用模型进行推理, 可用于数据处理
"""

import os, os.path as osp
import glob
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision
from args import Configs
from models import Net


preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 384)),
    torchvision.transforms.RandomCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
])


def get_img_list():
    root_path = '/home/sunnypc/dangxs/datasets/IQA/tid2013/distorted_images'
    root_path = '/home/sunnypc/dangxs/datasets/IQA/CSIQ/dst_imgs/blur'
    files = glob.glob(r"{}/*[jpg,jpeg,png,ppm,bmp,BMP,pgm,tif,tiff,webpv]".format(root_path), recursive=True)
    sorted(files)
    return files


@torch.no_grad()
def model_infer():
    pretrained_path = 'Save_Models_TReS/tid2013_1_2021/sv/bestmodel_1_2021.zip'
    assert osp.exists(pretrained_path)

    device = torch.device("cuda:0")
    config = Configs()
    # config.network = 'resnet18'
    # config.network = 'resnet34'
    config.network = 'resnet50'
    config.nheadt = 16
    config.num_encoder_layerst = 2
    config.dim_feedforwardt = 64

    net = Net(config, device, False).to(device)
    net.eval()
    print('model create is done.')

    net.load_state_dict(torch.load(pretrained_path))
    print('load checkpoint is done.')

    tmp_savepath = './infer_save/tid2013/distorted_images'
    tmp_savepath = './infer_save/CSIQ/dst_imgs/blur'

    img_list = get_img_list()
    assert len(img_list) > 0
    for f in img_list:
        img = cv2.imread(f)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_i = preprocess(img_pil)[None, ...]

        img_i = torch.as_tensor(img_i, dtype=torch.float32).to(device)
        predictionQA, _ = net(img_i)
        print(f, predictionQA)

        iqa_score = predictionQA[0].item()
        cv2.putText(img, 'IQA: {:.2f}'.format(iqa_score), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,255),2)

        _savepath = f'{tmp_savepath}/{osp.basename(f)}'
        os.makedirs(osp.dirname(_savepath), exist_ok=True)
        cv2.imwrite(_savepath, img)

    print('done.')

"""
python model_inference.py
"""

if __name__ == '__main__':
    model_infer()
