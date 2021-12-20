import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from Data_and_Loader.WLFW_Dataset import WLFWDatasets
from model_structure.Backbone_net import BackboneNet
from model_structure.auxiliary_net import Auxiliary_part
from Train_Test.lossfunc import PFLDLoss
from Train_Test.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))

def str2bool(string_info):
    if string_info.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string_info.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, plfd_backbone, auxiliarynet, criterion, optimizer, epoch):
    losses = AverageMeter()

    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        print(type(img))
        print(type(landmark_gt))
        print(type(attribute_gt))
        print(type(euler_angle_gt))
        img = img.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        plfd_backbone = plfd_backbone.to(device)
        auxiliarynet = auxiliarynet.to(device)
        features, landmarks = plfd_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, args.train_batchsize)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        losses.update(loss.item())
        print(loss)
    return weighted_loss, loss


def validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion):
    plfd_backbone.eval()
    auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            # attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            # euler_angle_gt = euler_angle_gt.to(device)
            plfd_backbone = plfd_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = plfd_backbone(img)
            # 只看关键点的损失
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)


def main(args):
    # 实例化骨干网络和辅助网络
    plfd_backbone = BackboneNet().to(device)
    auxiliarynet = Auxiliary_part().to(device)
    # 实例化损失，优化器等
    criterion = PFLDLoss()
    optimizer = torch.optim.Adam(
        [{'params': plfd_backbone.parameters()}, {'params': auxiliarynet.parameters()}], lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)
    # 加载数据并进行数据增强
    transform = transforms.Compose([transforms.ToTensor()])
    wlfwdataset = WLFWDatasets(args.dataroot, transform)
    print(type(wlfwdataset[0]))
    dataloader = DataLoader(
        wlfwdataset,
        batch_size=args.train_batchsize,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False)

    wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transform)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.workers)
    # 训练和验证网络，并记录断点，方便回溯
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, plfd_backbone, auxiliarynet, criterion, optimizer, epoch)
        filename = os.path.join(str(args.snapshot), "checkpoint_epoch_" + str(epoch) + '.pth.tar')
        save_checkpoint({'epoch': epoch,'plfd_backbone': plfd_backbone.state_dict(),'auxiliarynet': auxiliarynet.state_dict()}, filename)
        val_loss = validate(wlfw_val_dataloader, plfd_backbone, auxiliarynet, criterion)
        scheduler.step(val_loss)
        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {'val loss': val_loss, 'train loss': train_loss}, epoch)
    writer.close()

# 定义配置，可调节的参数
def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)
    parser.add_argument("--lr_patience", default=40, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--end_epoch', default=100, type=int)
    parser.add_argument('--snapshot',default='./checkpoint/snapshot/',type=str,metavar='PATH')
    parser.add_argument('--tensorboard', default="./checkpoint/tensorboard", type=str)
    # 训练数据集的记录TXT
    parser.add_argument('--dataroot',default='./data/train_data/list.txt',type=str,metavar='PATH')
    # 验证数据集的记录TXT
    parser.add_argument('--val_dataroot',default='./data/test_data/list.txt',type=str,metavar='PATH')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 参数实例化，如果不传入参数则按默认参数进行
    args = parse_args()
    # 运行
    main(args)
