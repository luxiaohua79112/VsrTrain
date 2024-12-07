import argparse
import os
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from espcn_model import EspcnNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='epoch_3_100.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    # 加载ESPCN模型
    model = EspcnNet(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('espcn_epochs/' + MODEL_NAME))  # 加载参数数据

    # 输出参数数据
    #for var_name in model.state_dict():
    #    print(var_name, '\t', model.state_dict()[var_name])

    # 创建输出图像目录
    out_path = 'espcn_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 加载图像数据，转换成 YCbCr格式
    # image_name = 'data/test/small.jpg'
    image_name = 'd:/sr/240x320.bmp'
    img = Image.open(image_name).convert('YCbCr')
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
  #  print('image.shape=', image.data.shape)
  #  print('image.shape=', image)
    if torch.cuda.is_available():
        image = image.cuda()

    model.outdata()
    print('upsample begin...')
    out = model(image)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_img.save(out_path + 'target_3x.bmp')
    print('upsample done!')