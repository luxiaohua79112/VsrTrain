import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch as F
from PIL import Image
import numpy as np
from torchvision import transforms

class EspcnNet(nn.Module):
    def __init__(self, upscale_factor):
        super(EspcnNet, self).__init__()

        # 输入1个通道，输出32通道，5*5卷积核大小，移动步长1像素，边缘填充2像素
        self.conv1 = nn.Conv2d(1, 32, (5, 5), (1, 1), (2, 2))

        # 输入32个通道，输出16通道，3*3卷积核大小，移动步长1像素，边缘填充1像素
        self.conv2 = nn.Conv2d(32, 16, (3, 3), (1, 1), (1, 1))

        # 输入16个通道，输出r^2个通道(放大3倍是9通道)，3*3卷积核大小，移动步长1像素，边缘填充1像素
        self.conv3 = nn.Conv2d(16, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))

        # 最后通过亚像素卷积操作
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        torch.set_printoptions(profile="full")
        torch.set_printoptions(precision=6)  # 显示浮点数的精度到小数点后6位

        # loader使用torchvision中自带的transforms函数
        self.loader = transforms.Compose([transforms.ToTensor()])
        self.unloader = transforms.ToPILImage()


    def outdata(self):
      #  print_options.set_float_precision(2)
        print('====================================================\n')
        print('conv1.bias.size = ', self.conv1.bias.data.shape)
        print('conv1.weight.size = ', self.conv1.weight.data.shape)

        print('\nconv1.bias=\n')
        print(self.conv1.bias)

        print('\nconv1.weight=\n')
        print(self.conv1.weight)

        print('====================================================\n')
        print('conv2.bias.size = ', self.conv2.bias.data.shape)
        print('conv2.weight.size = ', self.conv2.weight.data.shape)
        print('\nconv2.bias=\n')
        print(self.conv2.bias)

        print('\nconv2.weight=\n')
        print(self.conv2.weight)

        print('====================================================\n')
        print('conv3.bias.size = ', self.conv3.bias.data.shape)
        print('conv3.weight.size = ', self.conv3.weight.data.shape)

        print('\nconv3.bias=\n')
        print(self.conv3.bias)

        print('\nconv3.weight=\n')
        print(self.conv3.weight)

        print('====================================================\n\n')

    # 输入tensor变量
    # 输出PIL格式图片
    def tensor_to_pil(self, tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        return image

    def forward(self, x):
        # 第一层 tanh 激活
       # print('Convoluting 1 ......\n')
        x = F.tanh(self.conv1(x))
        folder_path1 = 'd:/sr/PyTorch/conv1/torch_conv1_'
        # self.print_conv_out(x, folder_path1)

        # 第二层 tanh 激活
       # print('Convoluting 2 ......\n')
        x = F.tanh(self.conv2(x))
        folder_path2 = 'd:/sr/PyTorch/conv2/torch_conv2_'
        # self.print_conv_out(x, folder_path2)

        # 第三层卷积（这里不需要激活）
       # print('Convoluting 3 ......\n')
        x = self.conv3(x)
        folder_path3 = 'd:/sr/PyTorch/conv3/torch_conv3_'
        # self.print_conv_out(x, folder_path3)

        # 亚像素卷积后，Sigmod激活
       # print('PixelShuffling ......\n')
        # x = F.sigmoid(self.pixel_shuffle(x))
        x = F.sigmoid(x)
        folder_path4 = 'd:/sr/PyTorch/conv4/torch_conv4_'
        # self.print_conv_out2(x, folder_path4)

        x = self.pixel_shuffle(x)

        print('forward done!\n')
        return x


if __name__ == "__main__":
    model = EspcnNet(upscale_factor=3)
    print(model)
