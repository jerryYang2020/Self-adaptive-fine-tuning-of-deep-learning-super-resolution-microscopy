import torch.nn as nn
import torch
from torch import autograd
import math
import torch.nn.functional as F
from torch.autograd import Variable

class DoubleConv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self,unet_size):
        a=unet_size
        super(Unet, self).__init__()
        in_ch,out_ch=1,1
        drop=0.5
        convnum=48
        self.conv1 = DoubleConv2(in_ch, convnum)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(DoubleConv2(convnum, convnum*2),nn.Dropout(drop))
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Sequential(DoubleConv2(convnum*2, convnum*4),nn.Dropout(drop))
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = nn.Sequential(DoubleConv2(convnum*4, convnum*8),nn.Dropout(drop))
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = nn.Sequential(DoubleConv2(convnum*8, convnum*16),nn.Dropout(drop))
        self.up6 = nn.Sequential(nn.Upsample(a),DoubleConv2(convnum*16, convnum*8),nn.Dropout(drop))
        self.conv6 = DoubleConv2(convnum*16, convnum*8)
        self.up7 = nn.Sequential(nn.Upsample([x*2 for x in a]),DoubleConv2(convnum*8, convnum*4),nn.Dropout(drop))
        self.conv7 = DoubleConv2(convnum*8, convnum*4)
        self.up8 = nn.Sequential(nn.Upsample([x*4 for x in a]),DoubleConv2(convnum*4, convnum*2),nn.Dropout(drop))
        self.conv8 = DoubleConv2(convnum*4, convnum*2)
        self.up9 = nn.Sequential(nn.Upsample([x*8 for x in a]),DoubleConv2(convnum*2, convnum),nn.Dropout(drop))
        self.conv9 = DoubleConv2(convnum*2, convnum)
        self.conv10 = nn.Conv2d(convnum, out_ch, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        c6 = torch.cat([up_6, c4],dim=1)
        c6 = self.conv6(c6)


        up_7 = self.up7(c6)
        c7 = torch.cat([up_7, c3],dim=1)
        c7 = self.conv7(c7)


        up_8 = self.up8(c7)
        c8 = torch.cat([up_8, c2],dim=1)
        c8 = self.conv8(c8)

        up_9 = self.up9(c8)
        c9 = torch.cat([up_9, c1],dim=1)
        c9 = self.conv9(c9)
        c10 = self.conv10(c9)



        return c10

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=8):
        super(RRDBNet, self).__init__()
        self.scale = scale

        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        if self.scale == 4:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        if self.scale == 8:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=4, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
