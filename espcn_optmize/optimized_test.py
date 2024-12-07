import argparse
import os
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_utils import is_image_file
from optimized_model import OptimzedNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='epoch_3_100.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    # 加载优化后的模型
    model = OptimzedNet(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('optimized_epochs/' + MODEL_NAME))  # 加载参数数据

    # 输出参数数据
    print('============ model parameter ============')
    model.outdata()
    # for var_name in model.state_dict():
    #     print(var_name, '\t', model.state_dict()[var_name])

    # 创建输出图像目录
    # out_path = 'd:/sr/result_optimized/'
    out_path = 'd:/Master/VideoSR/code/espcn_optmize/data/val/SRF_3/optimized/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 枚举要超分的源图像目录
    # path = 'd:/sr/testset/'
    path = 'd:/Master/VideoSR/code/espcn_optmize/data/val/SRF_3/data/'
    images_name = [x for x in listdir(path) if is_image_file(x)]

    # 依次遍历每一个原图像，进行超分放大处理
    for image_name in tqdm(images_name, desc='convert LR images to HR images'):
        img = Image.open(path + image_name).convert('YCbCr')
        y, cb, cr = img.split()
        image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        if torch.cuda.is_available():
            image = image.cuda()

        out = model(image)
        out = out.cpu()
        out_img_y = out.data[0].numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        out_img.save(out_path + image_name)
        print('upsample file: ' + image_name)

    # 所有图像超分处理完成
    print('upsample done!')