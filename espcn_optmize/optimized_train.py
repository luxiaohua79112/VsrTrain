import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from data_utils import DatasetFromFolder
from optimized_model import OptimzedNet
from psnrmeter import PSNRMeter

################################
# @brief 样本数据的处理函数
#################################
def processor(sample):
    data, target, training = sample
    data = Variable(data)
    target = Variable(target)
    if torch.cuda.is_available():  # 切换到GPU上训练
        data = data.cuda()
        target = target.cuda()

    output = model(data)              # 网络模型数据
    loss = criterion(output, target)  # 损失函数 使用均方误差

    return loss, output

################################
# @brief 往训练数据集中添加一个训练样本
#################################
def on_sample(state):
    state['sample'].append(state['train'])


################################
# @brief 清除训练数据
#################################
def reset_meters():
    meter_psnr.reset()
    meter_loss.reset()

################################
# @brief 前向预测处理，将结果添加到 psnr 和 损失计算 中
#################################
def on_forward(state):
    meter_psnr.add(state['output'].data, state['sample'][1])
#    meter_loss.add(state['loss'].data[0])
    meter_loss.add(state['loss'].item())

################################
# @brief 开始一次全样本训练
#################################
def on_start_epoch(state):
    reset_meters()
    scheduler.step()
    state['iterator'] = tqdm(state['iterator'])

################################
# @brief 完成一次全样本训练
#################################
def on_end_epoch(state):
    print('[Epoch %d] Train Loss: %.4f (PSNR: %.2f db)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    # 输出相关日志
    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_psnr_logger.log(state['epoch'], meter_psnr.value())

    reset_meters()  # 清除累计信息

    engine.test(processor, val_loader)
    val_loss_logger.log(state['epoch'], meter_loss.value()[0])
    val_psnr_logger.log(state['epoch'], meter_psnr.value())

    print('[Epoch %d] Val Loss: %.4f (PSNR: %.2f db)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    torch.save(model.state_dict(), 'optimized_epochs/epoch_%d_%d.pt' % (UPSCALE_FACTOR, state['epoch']))



#####################################################
# @brief 模型训练 入口主函数
#####################################################
if __name__ == "__main__":

    print('Trainning setting params......\n')
    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # 加载 训练数据集 和 测试数据集，加载后的图像数据转换为 [0.0, 1.0] 的向量
    print('Trainning loading data......\n')
    train_set = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False)

    print('Trainning loading Network......\n')
    model = OptimzedNet(upscale_factor=UPSCALE_FACTOR)
    criterion = nn.MSELoss()    # 使用均方误差 损失函数
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        print('Trainning use cuda......\n')

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    print('Trainning creating engine......\n')
    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()  # 求均值和标准差
    meter_psnr = PSNRMeter()

    print('Trainning VisdomPlot Logger......\n')
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': 'Val Loss'})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Val PSNR'})

    # 设置训练过程中的回调函数
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    # 开始训练数据集
    print('Trainning start......\n')
    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
    print('Trainning is done!\n')
