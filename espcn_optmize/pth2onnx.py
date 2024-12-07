import torch
import torchvision
import torch.nn as nn
import collections
import numpy as np
from data_utils import is_image_file
from optimized_model import OptimzedNet
import torch.onnx
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor

# from torchsummary import summary

#####################################################
# @brief 模型训练 入口主函数
#####################################################
if __name__ == "__main__":

    # 加载ESPCN模型
    model = OptimzedNet(upscale_factor=3)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('optimized_epochs/epoch_3_100.pt'))  # 加载参数数据
    model.eval()


    out_onnx = './optimized_model_3x.onnx'           #保存生成的onnx文件路径

    # 加载图像数据，转换成 YCbCr格式
    image_name = 'd:/sr/320x240.bmp'
    img = Image.open(image_name).convert('YCbCr')
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    if torch.cuda.is_available():
        image = image.cuda()

    print('Exporing begin...')
    out = model(image)
    input_names = ["input"]
    output_names = ["output"]
    torch_out = torch.onnx.export(model, image, out_onnx, keep_initializers_as_inputs=False, verbose=True,
                          input_names=input_names, output_names=output_names, opset_version=11)

    print('Exporing end!\n')

