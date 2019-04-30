import torch
import torch.nn as nn
#import model.ops as ops

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        ksize=3, stride=1, pad=1,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out)+x, inplace=True)
        return out


class EResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            3, 1, 1,
            groups=groups
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            3, 1, 1,
            groups=groups
        )
        self.pw = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.relu(self.pw(out)+x, inplace=True)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, groups=1):
        super().__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, groups=groups)
            self.up3 = _UpsampleBlock(n_channels, scale=3, groups=groups)
            self.up4 = _UpsampleBlock(n_channels, scale=4, groups=groups)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, groups=groups)
        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            if scale == 3:
                return self.up3(x)
            if scale == 4:
                return self.up4(x)
            raise NotImplementedError
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, groups=1):
        super().__init__()

        self.body = nn.ModuleList()
        if scale in [2, 4, 8]:
            for _ in range(int(math.log(scale, 2))):
                self.body.append(
                    nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=groups)
                )
                self.body.append(nn.ReLU(inplace=True))
                self.body.append(nn.PixelShuffle(2))
        elif scale == 3:
            self.body.append(
                nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=groups)
            )
            self.body.append(nn.ReLU(inplace=True))
            self.body.append(nn.PixelShuffle(3))

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        return out


class Block(nn.Module):
    def __init__(self, channel=64, mobile=False, groups=1):
        super().__init__()

        if mobile:
            self.b1 = EResidualBlock(channel, channel, groups=groups)
            self.b2 = self.b3 = self.b1
        else:
            self.b1 = ResidualBlock(channel, channel)
            self.b2 = ResidualBlock(channel, channel)
            self.b3 = ResidualBlock(channel, channel)
        self.c1 = nn.Conv2d(channel*2, channel, 1, 1, 0)
        self.c2 = nn.Conv2d(channel*3, channel, 1, 1, 0)
        self.c3 = nn.Conv2d(channel*4, channel, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class Net(nn.Module):
    def __init__(
        self,
        scale=4, multi_scale=True,
        num_channels=64,
        mobile=False, groups=1
    ):
        super().__init__()

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry = nn.Conv2d(3, num_channels, 3, 1, 1)

        self.b1 = Block(num_channels, mobile, groups)
        self.b2 = Block(num_channels, mobile, groups)
        self.b3 = Block(num_channels, mobile, groups)
        self.c1 = nn.Conv2d(num_channels*2, num_channels, 1, 1, 0)
        self.c2 = nn.Conv2d(num_channels*3, num_channels, 1, 1, 0)
        self.c3 = nn.Conv2d(num_channels*4, num_channels, 1, 1, 0)

        self.upsample = UpsampleBlock(
            num_channels,
            scale=scale, multi_scale=multi_scale,
            groups=groups
        )
        self.exit = nn.Conv2d(num_channels, 3, 3, 1, 1)

    def forward(self, x, scale=4):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        out = self.upsample(o3+x, scale=scale)

        out = self.exit(out)
        out = self.add_mean(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, downsample=1):
        super().__init__()

        def conv_bn_lrelu(in_channels, out_channels, ksize, stride, pad):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(),
            conv_bn_lrelu(64, 64, 4, 2, 1),
            conv_bn_lrelu(64, 128, 3, 1, 1),
            conv_bn_lrelu(128, 128, 4, 2, 1),
            conv_bn_lrelu(128, 256, 3, 1, 1),
            conv_bn_lrelu(256, 256, 4, 2, 1),
            conv_bn_lrelu(256, 512, 3, 1, 1),
            conv_bn_lrelu(512, 512, 3, 1, 1),
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        if downsample > 1:
            self.avg_pool = nn.AvgPool2d(downsample)

        self.downsample = downsample

    def forward(self, x):
        if self.downsample > 1:
            x = self.avg_pool(x)

        out = self.body(x)
        return out

def get_G(str1="carn_ganm",str2=""):


    from collections import OrderedDict
    import torch

    if str1=="carn_ganm":
        kwargs = {
            "num_channels": 64,
            "groups": 4,
            "mobile": True,
            "scale": 0,
        }
        ckpt= "/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/xx_model/PCARN-M-L1.pth"

    
    elif str1=="carn_gan":
        kwargs = {
            "num_channels": 64,
            "groups": 1,
            "mobile": False,
            "scale": 0,
        }
        ckpt= "/home/wangjian7/workspace/src/code_NNNNIPS/SR/wj_srfeat/sr_seaf/xx_model/PCARN-L1.pth"
    else :
        raise Exception("ss")

    net = Net(**kwargs)
    #'''
    state_dict = torch.load(ckpt)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    #'''
    return net

if __name__=="__main__":
    net = get_G()
    print (net)
    x=torch.randn(1,3,74,74)
    y=net(x,scale=4)
    print (y.shape)
