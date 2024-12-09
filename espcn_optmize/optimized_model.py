import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch as F
from PIL import Image
import numpy as np
from torchvision import transforms

class OptimzedNet(nn.Module):
    def __init__(self, upscale_factor):
        super(OptimzedNet, self).__init__()

        # 输入1个通道，输出36通道，5*5卷积核大小，移动步长1像素，边缘填充2像素
        self.conv1 = nn.Conv2d(1, 36, (5, 5), (1, 1), (2, 2))

        # 输入36个通道，输出18通道，3*3卷积核大小，移动步长1像素，边缘填充1像素，卷积膨胀1像素，分6个组
        self.conv2 = nn.Conv2d(36, 18, (3, 3), (1, 1), (1, 1), 1, 6)

        # 输入18个通道，输出r^2个通道(放大3倍是9通道)，3*3卷积核大小，移动步长1像素，边缘填充1像素, 卷积膨胀1像素，分3个组
        self.conv3 = nn.Conv2d(18, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1), 1, 3)

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


    # tensor 转换成 序列image图片输出
    def print_conv_out(self, x, folder):
        conv_out = x.cpu()
        print(conv_out.shape)
        conv_imgarray = conv_out[0]
        for j in range(conv_out.shape[1]):
            conv_mtx = conv_imgarray[j].detach().numpy()
            conv_mtx *= 128.0
            conv_mtx += 127.0
            conv_img = Image.fromarray(np.uint8(conv_mtx), mode='L')
            file_name = folder + str(j) + '.bmp'
            conv_img.save(file_name)
        return 0;

    # tensor 转换成 序列image图片输出
    def print_conv_out2(self, x, folder):
        conv_out = x.cpu()
        print(conv_out.shape)
        conv_imgarray = conv_out[0]
        for j in range(conv_out.shape[1]):
            conv_mtx = conv_imgarray[j].detach().numpy()
            conv_mtx *= 255.0
            conv_img = Image.fromarray(np.uint8(conv_mtx), mode='L')
            file_name = folder + str(j) + '.bmp'
            conv_img.save(file_name)
        return 0;


    #############################
    # @brief: 通道混淆操作
    #############################
    def channel_shuffle(self, x, groups:int):
        batchsize, num_channels, height, width = x.data.size()

        # 计算每组的通道数
        channels_per_group = num_channels // groups

        # Reshape操作，将通道扩展为两维
        x = x.view(batchsize, groups, channels_per_group, height, width)

        # Transpose操作，将组卷积两个维度进行置换
        x = torch.transpose(x, 1, 2).contiguous()

        # Flatten操作，两个维度平展成一个维度
        x = x.view(batchsize, -1, height, width)
        return x


    #############################
    # @brief: 前向推理过程
    #############################
    def forward(self, x):
        # 第一层 (5*5的卷积 + tanh激活)
        print('Convoluting 1 ......\n')
        x = self.conv1(x)
        x = F.tanh(x)
        folder_path1 = 'd:/sr/PyTorch/conv1/torch_conv1_'
#        self.print_conv_out(x, folder_path1)

        # 将第一层输出进行 channel shuffle
        x = self.channel_shuffle(x, 6)

        # 第二层 (3*3的卷积 + tanh激活)
        print('Convoluting 2 ......\n')
        x = self.conv2(x)
        x = F.tanh(x)
        folder_path2 = 'd:/sr/PyTorch/conv2/torch_conv2_'
#        self.print_conv_out(x, folder_path2)

        # 将第二层输出进行 channel shuffle
        x = self.channel_shuffle(x, 3)

        # 第三层 (3*3的卷积 + Sigmod激活)
        print('Convoluting 3 ......\n')
        x = self.conv3(x)
        x = F.sigmoid(x)
        folder_path3 = 'd:/sr/PyTorch/conv3/torch_conv3_'
#        self.print_conv_out(x, folder_path3)


        # 最后一层，(亚像素卷积)
        x = self.pixel_shuffle(x)

        print('forward done!\n')
        return x


if __name__ == "__main__":
    model = OptimzedNet(upscale_factor=2)
    print(model)
