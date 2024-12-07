import argparse
import os
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as peak_signal_noise_ratio
from data_utils import is_image_file
import matplotlib.pyplot as plt


if __name__ == "__main__":

    path_src1 = 'd:/Master/VideoSR/code/espcn_optmize/data/val/SRF_3/target/'
    path_src2 = 'd:/Master/VideoSR/code/espcn_optmize/data/val/SRF_3/optimized/'
    images_name = [x for x in listdir(path_src1) if is_image_file(x)]

    # 依次遍历每一个原图像
    print('============ PSNR OPTIMIZED BEGIN ================')
    index = 1
    x = []
    y1 = []
    for image_name in tqdm(images_name):
        # 读取原图
        image_src1 = io.imread(path_src1 + image_name)
        image_src2 = io.imread(path_src2 + image_name)

        # 求psnr值
        psnr_value = peak_signal_noise_ratio(image_src1, image_src2)
        print('fileName=' + image_name + ', psnr_value=', psnr_value)

        x.append(index)
        y1.append(psnr_value)
        index = index + 1

    # 所有PSNR处理完成
    print('============ PSNR OPTIMIZED END ================')


    path_src1 = 'd:/Master/VideoSR/code/espcn_optmize/data/val/SRF_3/target/'
    path_src2 = 'd:/Master/VideoSR/code/espcn_optmize/data/val/SRF_3/optimized/'
    images_name = [x for x in listdir(path_src1) if is_image_file(x)]

    # 依次遍历每一个原图像
    print('============ PSNR ESPCN BEGIN ================')
    index = 1
    x = []
    y2 = []
    for image_name in tqdm(images_name):
        # 读取原图
        image_src1 = io.imread(path_src1 + image_name)
        image_src2 = io.imread(path_src2 + image_name)

        # 求psnr值
        psnr_value = peak_signal_noise_ratio(image_src1, image_src2)
        print('fileName=' + image_name + ', psnr_value=', psnr_value)

        x.append(index)
        y2.append(psnr_value)
        index = index + 1

    # 所有PSNR处理完成
    print('============ PSNR ESPCN END ================')



  #  plt.plot(x, y1, color='blue', alpha=0.5, label='Optimized')
  #  plt.plot(x, y2, color='red', alpha=0.5, label='ESPCN')

    y3 = [i for i in y1 if i not in y2]

    # plt.plot(x, y3, 'o', color='black', alpha=1.0, label='ESPCN')
    plt.plot(x, y3, alpha=1.0, label='Optimized')
    plt.xlabel("Test DataSet")
    plt.ylabel("PSNR")

 #   fig, ax = plt.subplots()
 #   ax.plot(x, y3, color='gray', alpha=1.0, label='ESPCN')
 #   ax.set_xlabel('Test DataSet')
 #   ax.set_ylabel('PSNR')

    plt.legend()
    plt.show()