# created by Titas De
# basic structure borrowed from Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import pdb
import copy

__all__ = ['InceptionFourView', 'inception_fourview']


# model_urls = {
#     # Inception v3 ported from TensorFlow
#     'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
# }


def inception_fourview(pretrained=False, **kwargs):
    """Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    return InceptionTwoView(**kwargs)


class InceptionFourView(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, dropout_prob=0.5, aux_dropout=False):
        super(InceptionFourView, self).__init__()
#        torch.manual_seed(manual_seed)
#        print(torch.initial_seed())

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.Conv2d_1a_3x3_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3_1 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3_1 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1_1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3_1 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b_1 = InceptionA(192, pool_features=32)
        self.Mixed_5c_1 = InceptionA(256, pool_features=64)
        self.Mixed_5d_1 = InceptionA(288, pool_features=64)
        self.Mixed_6a_1 = InceptionB(288)
        self.Mixed_6b_1 = InceptionC(768, channels_7x7=128)
        self.Mixed_6c_1 = InceptionC(768, channels_7x7=160)
        self.Mixed_6d_1 = InceptionC(768, channels_7x7=160)
        self.Mixed_6e_1 = InceptionC(768, channels_7x7=192)

        self.Conv2d_1a_3x3_2 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3_2 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3_2 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1_2 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3_2 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b_2 = InceptionA(192, pool_features=32)
        self.Mixed_5c_2 = InceptionA(256, pool_features=64)
        self.Mixed_5d_2 = InceptionA(288, pool_features=64)
        self.Mixed_6a_2 = InceptionB(288)
        self.Mixed_6b_2 = InceptionC(768, channels_7x7=128)
        self.Mixed_6c_2 = InceptionC(768, channels_7x7=160)
        self.Mixed_6d_2 = InceptionC(768, channels_7x7=160)
        self.Mixed_6e_2 = InceptionC(768, channels_7x7=192)

        self.Conv2d_1a_3x3_3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3_3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3_3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1_3 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3_3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b_3 = InceptionA(192, pool_features=32)
        self.Mixed_5c_3 = InceptionA(256, pool_features=64)
        self.Mixed_5d_3 = InceptionA(288, pool_features=64)
        self.Mixed_6a_3 = InceptionB(288)
        self.Mixed_6b_3 = InceptionC(768, channels_7x7=128)
        self.Mixed_6c_3 = InceptionC(768, channels_7x7=160)
        self.Mixed_6d_3 = InceptionC(768, channels_7x7=160)
        self.Mixed_6e_3 = InceptionC(768, channels_7x7=192)

        self.Conv2d_1a_3x3_4 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3_4 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3_4 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1_4 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3_4 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b_4 = InceptionA(192, pool_features=32)
        self.Mixed_5c_4 = InceptionA(256, pool_features=64)
        self.Mixed_5d_4 = InceptionA(288, pool_features=64)
        self.Mixed_6a_4 = InceptionB(288)
        self.Mixed_6b_4 = InceptionC(768, channels_7x7=128)
        self.Mixed_6c_4 = InceptionC(768, channels_7x7=160)
        self.Mixed_6d_4 = InceptionC(768, channels_7x7=160)
        self.Mixed_6e_4 = InceptionC(768, channels_7x7=192)

        if aux_logits:
            self.AuxLogits_FourView = InceptionAux_FourView(768, num_classes, self.training, dropout_prob, aux_dropout)

        self.Mixed_7a_1 = InceptionD(768)
        self.Mixed_7b_1 = InceptionE(1280)
        self.Mixed_7c_1 = InceptionE(2048)

        self.Mixed_7a_2 = InceptionD(768)
        self.Mixed_7b_2 = InceptionE(1280)
        self.Mixed_7c_2 = InceptionE(2048)

        self.Mixed_7a_3 = InceptionD(768)
        self.Mixed_7b_3 = InceptionE(1280)
        self.Mixed_7c_3 = InceptionE(2048)

        self.Mixed_7a_4 = InceptionD(768)
        self.Mixed_7b_4 = InceptionE(1280)
        self.Mixed_7c_4 = InceptionE(2048)

        self.fc_fourview = nn.Linear(2048*4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_from_pretrained(self, model_pt):

        self.Conv2d_1a_3x3_1 = copy.deepcopy(model_pt.Conv2d_1a_3x3)
        self.Conv2d_2a_3x3_1 = copy.deepcopy(model_pt.Conv2d_2a_3x3)
        self.Conv2d_2b_3x3_1 = copy.deepcopy(model_pt.Conv2d_2b_3x3)
        self.Conv2d_3b_1x1_1 = copy.deepcopy(model_pt.Conv2d_3b_1x1)
        self.Conv2d_4a_3x3_1 = copy.deepcopy(model_pt.Conv2d_4a_3x3)
        self.Mixed_5b_1 = copy.deepcopy(model_pt.Mixed_5b)
        self.Mixed_5c_1 = copy.deepcopy(model_pt.Mixed_5c)
        self.Mixed_5d_1 = copy.deepcopy(model_pt.Mixed_5d)
        self.Mixed_6a_1 = copy.deepcopy(model_pt.Mixed_6a)
        self.Mixed_6b_1 = copy.deepcopy(model_pt.Mixed_6b)
        self.Mixed_6c_1 = copy.deepcopy(model_pt.Mixed_6c)
        self.Mixed_6d_1 = copy.deepcopy(model_pt.Mixed_6d)
        self.Mixed_6e_1 = copy.deepcopy(model_pt.Mixed_6e)

        self.Conv2d_1a_3x3_2 = copy.deepcopy(model_pt.Conv2d_1a_3x3)
        self.Conv2d_2a_3x3_2 = copy.deepcopy(model_pt.Conv2d_2a_3x3)
        self.Conv2d_2b_3x3_2 = copy.deepcopy(model_pt.Conv2d_2b_3x3)
        self.Conv2d_3b_1x1_2 = copy.deepcopy(model_pt.Conv2d_3b_1x1)
        self.Conv2d_4a_3x3_2 = copy.deepcopy(model_pt.Conv2d_4a_3x3)
        self.Mixed_5b_2 = copy.deepcopy(model_pt.Mixed_5b)
        self.Mixed_5c_2 = copy.deepcopy(model_pt.Mixed_5c)
        self.Mixed_5d_2 = copy.deepcopy(model_pt.Mixed_5d)
        self.Mixed_6a_2 = copy.deepcopy(model_pt.Mixed_6a)
        self.Mixed_6b_2 = copy.deepcopy(model_pt.Mixed_6b)
        self.Mixed_6c_2 = copy.deepcopy(model_pt.Mixed_6c)
        self.Mixed_6d_2 = copy.deepcopy(model_pt.Mixed_6d)
        self.Mixed_6e_2 = copy.deepcopy(model_pt.Mixed_6e)

        self.Conv2d_1a_3x3_3 = copy.deepcopy(model_pt.Conv2d_1a_3x3)
        self.Conv2d_2a_3x3_3 = copy.deepcopy(model_pt.Conv2d_2a_3x3)
        self.Conv2d_2b_3x3_3 = copy.deepcopy(model_pt.Conv2d_2b_3x3)
        self.Conv2d_3b_1x1_3 = copy.deepcopy(model_pt.Conv2d_3b_1x1)
        self.Conv2d_4a_3x3_3 = copy.deepcopy(model_pt.Conv2d_4a_3x3)
        self.Mixed_5b_3 = copy.deepcopy(model_pt.Mixed_5b)
        self.Mixed_5c_3 = copy.deepcopy(model_pt.Mixed_5c)
        self.Mixed_5d_3 = copy.deepcopy(model_pt.Mixed_5d)
        self.Mixed_6a_3 = copy.deepcopy(model_pt.Mixed_6a)
        self.Mixed_6b_3 = copy.deepcopy(model_pt.Mixed_6b)
        self.Mixed_6c_3 = copy.deepcopy(model_pt.Mixed_6c)
        self.Mixed_6d_3 = copy.deepcopy(model_pt.Mixed_6d)
        self.Mixed_6e_3 = copy.deepcopy(model_pt.Mixed_6e)

        self.Conv2d_1a_3x3_4 = copy.deepcopy(model_pt.Conv2d_1a_3x3)
        self.Conv2d_2a_3x3_4 = copy.deepcopy(model_pt.Conv2d_2a_3x3)
        self.Conv2d_2b_3x3_4 = copy.deepcopy(model_pt.Conv2d_2b_3x3)
        self.Conv2d_3b_1x1_4 = copy.deepcopy(model_pt.Conv2d_3b_1x1)
        self.Conv2d_4a_3x3_4 = copy.deepcopy(model_pt.Conv2d_4a_3x3)
        self.Mixed_5b_4 = copy.deepcopy(model_pt.Mixed_5b)
        self.Mixed_5c_4 = copy.deepcopy(model_pt.Mixed_5c)
        self.Mixed_5d_4 = copy.deepcopy(model_pt.Mixed_5d)
        self.Mixed_6a_4 = copy.deepcopy(model_pt.Mixed_6a)
        self.Mixed_6b_4 = copy.deepcopy(model_pt.Mixed_6b)
        self.Mixed_6c_4 = copy.deepcopy(model_pt.Mixed_6c)
        self.Mixed_6d_4 = copy.deepcopy(model_pt.Mixed_6d)
        self.Mixed_6e_4 = copy.deepcopy(model_pt.Mixed_6e)

        if self.aux_logits:
            self.AuxLogits_FourView.conv0_1 = copy.deepcopy(model_pt.AuxLogits.conv0)
            self.AuxLogits_FourView.conv1_1 = copy.deepcopy(model_pt.AuxLogits.conv1)
            self.AuxLogits_FourView.conv1_1.stddev = copy.deepcopy(model_pt.AuxLogits.conv1.stddev)
            self.AuxLogits_FourView.conv0_2 = copy.deepcopy(model_pt.AuxLogits.conv0)
            self.AuxLogits_FourView.conv1_2 = copy.deepcopy(model_pt.AuxLogits.conv1)
            self.AuxLogits_FourView.conv1_2.stddev = copy.deepcopy(model_pt.AuxLogits.conv1.stddev)
            self.AuxLogits_FourView.conv0_3 = copy.deepcopy(model_pt.AuxLogits.conv0)
            self.AuxLogits_FourView.conv1_3 = copy.deepcopy(model_pt.AuxLogits.conv1)
            self.AuxLogits_FourView.conv1_3.stddev = copy.deepcopy(model_pt.AuxLogits.conv1.stddev)
            self.AuxLogits_FourView.conv0_4 = copy.deepcopy(model_pt.AuxLogits.conv0)
            self.AuxLogits_FourView.conv1_4 = copy.deepcopy(model_pt.AuxLogits.conv1)
            self.AuxLogits_FourView.conv1_4.stddev = copy.deepcopy(model_pt.AuxLogits.conv1.stddev)
            self.AuxLogits_FourView.fc_fourview.stddev = copy.deepcopy(model_pt.AuxLogits.fc.stddev)

    def forward(self, x1, x2):
        if self.transform_input:
            x1 = x1.clone()
            x1[:, 0] = x1[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x1[:, 1] = x1[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x1[:, 2] = x1[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x2 = x2.clone()
            x2[:, 0] = x2[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x2[:, 1] = x2[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x2[:, 2] = x2[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x3 = x3.clone()
            x3[:, 0] = x3[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x3[:, 1] = x3[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x3[:, 2] = x3[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x4 = x4.clone()
            x4[:, 0] = x4[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x4[:, 1] = x4[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x4[:, 2] = x4[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        # 299 x 299 x 3
        x1 = self.Conv2d_1a_3x3_1(x1)
        # 149 x 149 x 32
        x1 = self.Conv2d_2a_3x3_1(x1)
        # 147 x 147 x 32
        x1 = self.Conv2d_2b_3x3_1(x1)
        # 147 x 147 x 64
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x1 = self.Conv2d_3b_1x1_1(x1)
        # 73 x 73 x 80
        x1 = self.Conv2d_4a_3x3_1(x1)
        # 71 x 71 x 192
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x1 = self.Mixed_5b_1(x1)
        # 35 x 35 x 256
        x1 = self.Mixed_5c_1(x1)
        # 35 x 35 x 288
        x1 = self.Mixed_5d_1(x1)
        # 35 x 35 x 288
        x1 = self.Mixed_6a_1(x1)
        # 17 x 17 x 768
        x1 = self.Mixed_6b_1(x1)
        # 17 x 17 x 768
        x1 = self.Mixed_6c_1(x1)
        # 17 x 17 x 768
        x1 = self.Mixed_6d_1(x1)
        # 17 x 17 x 768
        x1 = self.Mixed_6e_1(x1)

        # 299 x 299 x 3
        x2 = self.Conv2d_1a_3x3_2(x2)
        # 149 x 149 x 32
        x2 = self.Conv2d_2a_3x3_2(x2)
        # 147 x 147 x 32
        x2 = self.Conv2d_2b_3x3_2(x2)
        # 147 x 147 x 64
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x2 = self.Conv2d_3b_1x1_2(x2)
        # 73 x 73 x 80
        x2 = self.Conv2d_4a_3x3_2(x2)
        # 71 x 71 x 192
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x2 = self.Mixed_5b_2(x2)
        # 35 x 35 x 256
        x2 = self.Mixed_5c_2(x2)
        # 35 x 35 x 288
        x2 = self.Mixed_5d_2(x2)
        # 35 x 35 x 288
        x2 = self.Mixed_6a_2(x2)
        # 17 x 17 x 768
        x2 = self.Mixed_6b_2(x2)
        # 17 x 17 x 768
        x2 = self.Mixed_6c_2(x2)
        # 17 x 17 x 768
        x2 = self.Mixed_6d_2(x2)
        # 17 x 17 x 768
        x2 = self.Mixed_6e_2(x2)
        # 17 x 17 x 768

        # 299 x 299 x 3
        x3 = self.Conv2d_1a_3x3_1(x3)
        # 149 x 149 x 32
        x3 = self.Conv2d_2a_3x3_1(x3)
        # 147 x 147 x 32
        x3 = self.Conv2d_2b_3x3_1(x3)
        # 147 x 147 x 64
        x3 = F.max_pool2d(x3, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x3 = self.Conv2d_3b_1x1_1(x3)
        # 73 x 73 x 80
        x3 = self.Conv2d_4a_3x3_1(x3)
        # 71 x 71 x 192
        x3 = F.max_pool2d(x3, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x3 = self.Mixed_5b_1(x3)
        # 35 x 35 x 256
        x3 = self.Mixed_5c_1(x3)
        # 35 x 35 x 288
        x3 = self.Mixed_5d_1(x3)
        # 35 x 35 x 288
        x3 = self.Mixed_6a_1(x3)
        # 17 x 17 x 768
        x3 = self.Mixed_6b_1(x3)
        # 17 x 17 x 768
        x3 = self.Mixed_6c_1(x3)
        # 17 x 17 x 768
        x3 = self.Mixed_6d_1(x3)
        # 17 x 17 x 768
        x3 = self.Mixed_6e_1(x3)

        # 299 x 299 x 3
        x4 = self.Conv2d_1a_3x3_2(x4)
        # 149 x 149 x 32
        x4 = self.Conv2d_2a_3x3_2(x4)
        # 147 x 147 x 32
        x4 = self.Conv2d_2b_3x3_2(x4)
        # 147 x 147 x 64
        x4 = F.max_pool2d(x4, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x4 = self.Conv2d_3b_1x1_2(x4)
        # 73 x 73 x 80
        x4 = self.Conv2d_4a_3x3_2(x4)
        # 71 x 71 x 192
        x4 = F.max_pool2d(x4, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x4 = self.Mixed_5b_2(x4)
        # 35 x 35 x 256
        x4 = self.Mixed_5c_2(x4)
        # 35 x 35 x 288
        x4 = self.Mixed_5d_2(x4)
        # 35 x 35 x 288
        x4 = self.Mixed_6a_2(x4)
        # 17 x 17 x 768
        x4 = self.Mixed_6b_2(x4)
        # 17 x 17 x 768
        x4 = self.Mixed_6c_2(x4)
        # 17 x 17 x 768
        x4 = self.Mixed_6d_2(x4)
        # 17 x 17 x 768
        x4 = self.Mixed_6e_2(x4)
        # 17 x 17 x 768

        if self.training and self.aux_logits:
            aux = self.AuxLogits_FourView(x1,x2,x3,x4)
        # 17 x 17 x 768

        x1 = self.Mixed_7a_1(x1)
        # 8 x 8 x 1280
        x1 = self.Mixed_7b_1(x1)
        # 8 x 8 x 2048
        x1 = self.Mixed_7c_1(x1)
        # 8 x 8 x 2048
        x1 = F.avg_pool2d(x1, kernel_size=8)
        # # 1 x 1 x 2048
        x1 = F.dropout(x1, training=self.training)
        # 1 x 1 x 2048
        x1 = x1.view(x1.size(0), -1)
        # 2048

        # 17 x 17 x 768
        x2 = self.Mixed_7a_2(x2)
        # 8 x 8 x 1280
        x2 = self.Mixed_7b_2(x2)
        # 8 x 8 x 2048
        x2 = self.Mixed_7c_2(x2)
        # 8 x 8 x 2048
        x2 = F.avg_pool2d(x2, kernel_size=8)
        # # 1 x 1 x 2048
        x2 = F.dropout(x2, training=self.training)
        # 1 x 1 x 2048
        x2 = x2.view(x2.size(0), -1)
        # 2048

        x3 = self.Mixed_7a_1(x3)
        # 8 x 8 x 1280
        x3 = self.Mixed_7b_1(x3)
        # 8 x 8 x 2048
        x3 = self.Mixed_7c_1(x3)
        # 8 x 8 x 2048
        x3 = F.avg_pool2d(x3, kernel_size=8)
        # # 1 x 1 x 2048
        x3 = F.dropout(x3, training=self.training)
        # 1 x 1 x 2048
        x3 = x3.view(x3.size(0), -1)
        # 2048

        # 17 x 17 x 768
        x4 = self.Mixed_7a_2(x4)
        # 8 x 8 x 1280
        x4 = self.Mixed_7b_2(x4)
        # 8 x 8 x 2048
        x4 = self.Mixed_7c_2(x4)
        # 8 x 8 x 2048
        x4 = F.avg_pool2d(x4, kernel_size=8)
        # # 1 x 1 x 2048
        x4 = F.dropout(x4, training=self.training)
        # 1 x 1 x 2048
        x4 = x4.view(x4.size(0), -1)
        # 2048

        x = self.fc_fourview(torch.cat((x1,x2,x3,x4),dim=1))
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x



class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

'''
class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x
'''

class InceptionAux_FourView(nn.Module):

    def __init__(self, in_channels, num_classes, training=False, dropout_prob=1.0, aux_dropout=False):
        super(InceptionAux_FourView, self).__init__()
        self.conv0_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1_1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1_1.stddev = 0.01
        self.conv0_2 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1_2 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1_2.stddev = 0.01
        self.conv0_3 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1_3 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1_3.stddev = 0.01
        self.conv0_4 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1_4 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1_4.stddev = 0.01
        self.fc_fourview = nn.Linear(768*4, num_classes)
        self.fc_fourview.stddev = 0.001
        self.training = training
        self.dropout_prob = dropout_prob
        self.aux_dropout = aux_dropout

    def forward(self, x1, x2, x3, x4):

        # 17 x 17 x 768
        x1 = F.avg_pool2d(x1, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x1 = self.conv0_1(x1)
        # 5 x 5 x 128
        x1 = self.conv1_1(x1)
        # 1 x 1 x 768
        if self.aux_dropout:
            x1 = F.dropout(x1, training=self.training, p = self.dropout_prob)
        x1 = x1.view(x1.size(0), -1)
        # 768

        # 17 x 17 x 768
        x2 = F.avg_pool2d(x2, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x2 = self.conv0_2(x2)
        # 5 x 5 x 128
        x2 = self.conv1_2(x2)
        # 1 x 1 x 768
        if self.aux_dropout:
            x2 = F.dropout(x2, training=self.training, p = self.dropout_prob)
        x2 = x2.view(x2.size(0), -1)
        # 768

        # 17 x 17 x 768
        x3 = F.avg_pool2d(x3, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x3 = self.conv0_1(x3)
        # 5 x 5 x 128
        x3 = self.conv1_1(x3)
        # 1 x 1 x 768
        if self.aux_dropout:
            x3 = F.dropout(x3, training=self.training, p = self.dropout_prob)
        x3 = x3.view(x3.size(0), -1)
        # 768

        # 17 x 17 x 768
        x4 = F.avg_pool2d(x4, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x4 = self.conv0_2(x4)
        # 5 x 5 x 128
        x4 = self.conv1_2(x4)
        # 1 x 1 x 768
        if self.aux_dropout:
            x4 = F.dropout(x4, training=self.training, p = self.dropout_prob)
        x4 = x4.view(x4.size(0), -1)
        # 768

        x = torch.cat((x1,x2,x3,x4),dim=1)
        x = self.fc_fourview(x)
        # 1000
        return x



class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)