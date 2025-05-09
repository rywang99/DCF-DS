'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        #self.se = SELayer(planes * 4, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=40, embed_dim=128, squeeze_excitation=False):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.squeeze_excitation = squeeze_excitation
        if block is BasicBlock:
            self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
            current_freq_dim = int((feat_dim - 1) / 2) + 1
            self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.embedding = nn.Linear(m_channels * 8 * 2 * current_freq_dim, embed_dim)
        elif block is Bottleneck:
            self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
            self.embedding = nn.Linear(int(feat_dim/8) * m_channels * 16 * block.expansion, embed_dim)
        else:
            raise ValueError(f'Unexpected class {type(block)}.')
           

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, seg_len=100000):
        x = x.unsqueeze_(1)
        slen = x.shape[-1]
        embedding_all = 0
        N = 0
        for start in range(0, slen, seg_len):
            data = x[..., start:start + seg_len]
            out = F.relu(self.bn1(self.conv1(data)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            n = out.shape[-1]
            print(out.shape)
            pooling_mean = torch.mean(out, dim=-1)
            meansq = torch.mean(out * out, dim=-1)
            pooling_std = torch.sqrt(meansq - pooling_mean ** 2 + 1e-10)
            out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                            torch.flatten(pooling_std, start_dim=1)), 1)

            embedding = self.embedding(out)
            embedding_all = (N * embedding_all + n * embedding) / (N + n)
            N = N + n
        return embedding_all

class ResNet_SSP(nn.Module):
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=40, embed_dim=128, squeeze_excitation=False):
        super(ResNet_SSP, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.squeeze_excitation = squeeze_excitation
        if block is BasicBlock:
            self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
            current_freq_dim = int((feat_dim - 1) / 2) + 1
            self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
            current_freq_dim = int((current_freq_dim - 1) / 2) + 1
            #self.embedding = nn.Linear(m_channels * 8 * 2 * current_freq_dim, embed_dim)
        elif block is Bottleneck:
            self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(m_channels)
            self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)
            #self.embedding = nn.Linear(int(feat_dim/8) * m_channels * 16 * block.expansion, embed_dim)
        else:
            raise ValueError(f'Unexpected class {type(block)}.')
           

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, nT_len=10, nT_shift=5):
        x = x.unsqueeze_(1)
        data = x
        out = F.relu(self.bn1(self.conv1(data)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        T = out.shape[-1]
        final_out = []
        for start in range(0, T, nT_shift):
            if start + nT_len > T:
                continue
            data = out[..., start: start + nT_len]
            # print(data.shape, start)
            data = data.reshape(data.shape[0], data.shape[1], -1)
            counts = data.shape[2]
            pooling_mean = torch.mean(data, dim=2, keepdim=True)
            pooling_var = torch.sum((data - pooling_mean)**2, dim=2, keepdim=True) / counts
            
            pooling_std = torch.sqrt(pooling_var.clamp(min=1e-10))

            cat_out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                            torch.flatten(pooling_std, start_dim=1)), 1)
            final_out.append(cat_out)
            # print(cat_out.shape) 
        final_out = torch.stack(final_out, dim=1)
        # print(final_out.shape)
        
        return final_out

def ResNet_34(feat_dim, embed_dim, squeeze_excitation=False):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim, squeeze_excitation=squeeze_excitation)

def ResNet_SSP_34(feat_dim, embed_dim, squeeze_excitation=False):
    return ResNet_SSP(BasicBlock, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim, squeeze_excitation=squeeze_excitation)

def ResNet101(feat_dim, embed_dim, squeeze_excitation=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], feat_dim=feat_dim, embed_dim=embed_dim, squeeze_excitation=squeeze_excitation)

if __name__=='__main__':
    net = ResNet_34(80, 128)
    data = torch.randn(2, 80, 800) # fbank
    out = net(data)
    print(out.shape)