# created by Titas De
# need some more verification

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['inception_v4', 'inception_v4']

model_urls = {
    'inception_v4': 'https://s3.amazonaws.com/pytorch/models/inception_v4-58153ba9.pth'
}

def inception_v4(pretrained=False):
    r"""inception_v4 model architecture from the
    `"Inception-v4..." <https://arxiv.org/abs/1602.07261>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = inception_v4()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v4']))
    return model


class inception_v4(nn.Module):

    def __init__(self, num_classes=1001):
        super(inception_v4, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed3a(),
            Mixed4a(),
            Mixed5a(),
            InceptionA(),
            InceptionA(),
            InceptionA(),
            InceptionA(),
            ReductionA(), # Mixed6a
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            InceptionB(),
            ReductionB(), # Mixed7a
            InceptionC(),
            InceptionC(),
            InceptionC(),
            nn.AvgPool2d(8, count_include_pad=False)
        )
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x) 
        return x


class InceptionA(nn.Module):

    def __init__(self):
        super(InceptionA, self).__init__()
        self.chunk0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.chunk1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.chunk2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.chunk3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.chunk0(x)
        x1 = self.chunk1(x)
        x2 = self.chunk2(x)
        x3 = self.chunk3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionB(nn.Module):

    def __init__(self):
        super(InceptionB, self).__init__()
        self.chunk0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        
        self.chunk1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.chunk2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.chunk3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.chunk0(x)
        x1 = self.chunk1(x)
        x2 = self.chunk2(x)
        x3 = self.chunk3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionC(nn.Module):

    def __init__(self):
        super(InceptionC, self).__init__()
        self.chunk0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        
        self.chunk1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.chunk1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.chunk1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.chunk2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.chunk2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.chunk2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.chunk2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.chunk2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.chunk3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.chunk0(x)
        
        x1_0 = self.chunk1_0(x)
        x1_1a = self.chunk1_1a(x1_0)
        x1_1b = self.chunk1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.chunk2_0(x)
        x2_1 = self.chunk2_1(x2_0)
        x2_2 = self.chunk2_2(x2_1)
        x2_3a = self.chunk2_3a(x2_2)
        x2_3b = self.chunk2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.chunk3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class ReductionA(nn.Module):

    def __init__(self):
        super(ReductionA, self).__init__()
        self.chunk0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.chunk1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )
        
        self.chunk2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.chunk0(x)
        x1 = self.chunk1(x)
        x2 = self.chunk2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class ReductionB(nn.Module):

    def __init__(self):
        super(ReductionB, self).__init__()

        self.chunk0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.chunk1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.chunk2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.chunk0(x)
        x1 = self.chunk1(x)
        x2 = self.chunk2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) 
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed3a(nn.Module):

    def __init__(self):
        super(Mixed3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed4a(nn.Module):

    def __init__(self):
        super(Mixed4a, self).__init__()

        self.chunk0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.chunk1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

    def forward(self, x):
        x0 = self.chunk0(x)
        x1 = self.chunk1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed5a(nn.Module):

    def __init__(self):
        super(Mixed5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out










