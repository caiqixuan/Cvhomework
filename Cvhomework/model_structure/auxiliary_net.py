import torch
import torch.nn as nn

def ConvBNRelu(in_channel, out_channel, kernel, stride, padding=1,groups=1,bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6())

class Auxiliary_part(nn.Module):
    def __init__(self):
        super(Auxiliary_part, self).__init__()
        self.conv1 = ConvBNRelu(64, 128, 3, 2)
        self.conv2 = ConvBNRelu(128, 128, 3, 1)
        self.conv3 = ConvBNRelu(128, 32, 3, 2)
        self.conv4 = ConvBNRelu(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    input = torch.randn(1, 64, 28, 28)
    auxiliarynet = Auxiliary_part()
    print(auxiliarynet)
    angle = auxiliarynet(input)
    print("angle.shape:{0:}".format(angle.shape))