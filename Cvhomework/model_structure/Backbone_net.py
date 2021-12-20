from typing import Optional, List
import torch.nn as nn
import torch

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def ConvBNRelu(in_channel, out_channel, kernel, stride, padding=1,groups=1,bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6())

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio,use_c=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        if use_c:
            if stride == 1:
                self.shortcut = True
            else:
                self.shortcut = False
        else:
            self.shortcut = False
        self.inblock = nn.Sequential(
            ConvBNRelu(in_channel, in_channel * expand_ratio, 1, 1, 0),
            ConvBNRelu(in_channel * expand_ratio, in_channel * expand_ratio, 3, stride, 1, in_channel * expand_ratio),

            nn.Conv2d(in_channel * expand_ratio, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        if self.shortcut:
            y = self.inblock(x)
            return x + y
        else:
            return self.inblock(x)


class BackboneNet(nn.Module):
    def __init__(self,width_mult: float = 1.0, round_nearest: int = 8,):
        super(BackboneNet, self).__init__()
        # def ConvBNRelu(in_channel, out_channel, kernel, stride, padding=1,groups=1,bias=False)
        self.conv1 = ConvBNRelu(3,64,3,2,1)
        self.conv2 = ConvBNRelu(64,64,3,1,1)
        input_channel = 64
        feature_before_setting = [
            # t, c, n, s
            [2, 64, 5, 2],
        ]
        first_features: List[nn.Module] = []
        for t, c, n, s in feature_before_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                first_features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        feature_after_setting = [
            # t, c, n, s
            [2, 128, 1, 2],
            [4, 128, 6, 1],
            [2, 16, 1, 1],
        ]
        temp = 0
        second_features: List[nn.Module] = []
        for t, c, n, s in feature_after_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                temp += 1
                stride = s if i == 0 else 1
                # 第二阶段的第二层和最后一层没有shortcut
                if temp == 2 or temp ==8:
                    second_features.append(InvertedResidual(input_channel, output_channel, stride, t, use_c=False))
                else:
                    second_features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel


        self.features1 = nn.Sequential(*first_features)
        self.features2 = nn.Sequential(*second_features)
        self.conv_s1 = ConvBNRelu(16, 32, 3, 2)  # [32, 7, 7]
        self.conv_s2 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.relu6 = nn.ReLU6()
        # self.fBN = nn.BatchNorm2d(128)
        # 平均池化来对齐输入尺寸
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        # 这里的176=16+32+128 是s1 s2 s3 cat起来的， 196=2*98，代表98个点的x坐标和y坐标
        self.fc = nn.Linear(176, 196)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_to_auxiliary = self.features1(x)
        # print(x.shape)
        x = self.features2(x_to_auxiliary)
        # print(x.shape)
        s1 = x
        s2 = self.conv_s1(s1)
        s3 = self.relu6(self.conv_s2(s2))

        # 三个输入cat，然后通过卷积层
        s1 = self.avg_pool1(s1)
        s1 = s1.view(s1.size(0), -1)
        # print(s1.shape)
        s2 = self.avg_pool2(s2)
        s2 = s2.view(s2.size(0), -1)
        # print(s2.shape)

        s3 = s3.view(s1.size(0), -1)
        # print(s3.shape)
        multi_scale = torch.cat([s1, s2, s3], 1)
        Backboneout = self.fc(multi_scale)

        return x_to_auxiliary, Backboneout

if __name__ == '__main__':
    input = torch.randn(1, 3, 112, 112)
    plfd_backbone = BackboneNet()
    print(plfd_backbone)
    features, landmarks = plfd_backbone(input)

    print("landmarks.shape: {0:}, to_auxiliary.shape: {1:}".format(
         landmarks.shape,features.shape))