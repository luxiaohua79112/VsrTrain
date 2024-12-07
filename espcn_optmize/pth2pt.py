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
from torch.utils.mobile_optimizer import optimize_for_mobile

# from torchsummary import summary

#####################################################
# @brief 模型训练 入口主函数
#####################################################
if __name__ == "__main__":

    # 加载ESPCN模型
    model = OptimzedNet(upscale_factor=3)

    # 部署到手机上不要使用CUDA
#    if torch.cuda.is_available():
#        model = model.cuda()
    pre_weights = torch.load('optimized_epochs/epoch_3_100.pt')  # 读取参数
    model.load_state_dict(pre_weights, strict=True)  # 将参数载入到模型
    model.eval()  # 将模型设为验证模式

    out_pt = './optimized_model_3x.pt'           #保存生成的pt文件路径

    # 加载图像数据，转换成 YCbCr格式
    image_name = 'd:/sr/320x240.bmp'
    img = Image.open(image_name).convert('YCbCr')
    y, cb, cr = img.split()
    image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    # if torch.cuda.is_available():
    #    image = image.cuda()

    print('Exporing begin...')

    # 上面是准备模型，下面就是转换了

    # 用torch.jit.script转torchscript，不要用torch.jit.trace
    traced_script_module = torch.jit.script(model, image)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(out_pt)

    print('Exporing end!\n')

