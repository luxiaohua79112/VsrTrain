import argparse
import os
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
#from torchvision.transforms import Compose, CenterCrop, Scale, Resize
from torchvision.transforms import Compose, CenterCrop, Resize
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.bmp'])


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    scale_size = crop_size // upscale_factor
    return Compose([
        CenterCrop(crop_size),     # 居中裁剪
        Resize(scale_size, interpolation=Image.BICUBIC)  # 放大
      #  Scale(scale_size, interpolation=Image.BICUBIC)
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])

#####################################################
# @brief 从文件夹加载数据集 类
#        原始图像裁剪成 85*85大小的LR，然后放到factor倍 (3倍的话是 255*255大小)作为HR
#        LR 图像存放在  /data/train/SRF_3/data 目录下
#        HR 图像存放在  /data/train/SRF_3/target 目录下
#####################################################
class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/data'
        self.target_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/target'
        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir, x) for x in listdir(self.target_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, _, _ = Image.open(self.image_filenames[index]).convert('YCbCr').split()
        target, _, _ = Image.open(self.target_filenames[index]).convert('YCbCr').split()
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.image_filenames)


#####################################################
# @brief 通过数据集生成相应的训练数据
#        原始图像裁剪成 85*85大小的LR，然后放到factor倍 (3倍的话是 255*255大小)作为HR
#        LR 图像存放在  /data/train/SRF_3/data 目录下
#        HR 图像存放在  /data/train/SRF_3/target 目录下
#####################################################
def generate_dataset(data_type, upscale_factor):
    images_name = [x for x in listdir('data/VOC2012/' + data_type) if is_image_file(x)]
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    lr_transform = input_transform(crop_size, upscale_factor)
    hr_transform = target_transform(crop_size)

    root = 'data/' + data_type
    if not os.path.exists(root):
        os.makedirs(root)
    path = root + '/SRF_' + str(upscale_factor)
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = path + '/data'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    target_path = path + '/target'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for image_name in tqdm(images_name, desc='generate ' + data_type + ' dataset with upscale factor = '
            + str(upscale_factor) + ' from VOC2012'):
        image = Image.open('data/VOC2012/' + data_type + '/' + image_name)
        target = image.copy()
        image = lr_transform(image)
        target = hr_transform(target)

        image.save(image_path + '/' + image_name)
        target.save(target_path + '/' + image_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Super Resolution Dataset')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    print('================ data utils BEGIN ====================\n')

    # 生成训练数据集
    generate_dataset(data_type='train', upscale_factor=UPSCALE_FACTOR)

    # 生成验证数据集 (用来防止过拟合的)
    generate_dataset(data_type='val', upscale_factor=UPSCALE_FACTOR)

    print('================ data utils END ====================\n')
