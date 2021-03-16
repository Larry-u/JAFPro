import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import sys
sys.path.append("..")

from .flownet2_pytorch.networks.FlowNetSD import FlowNetSD


def vgg_preprocess(x):
    x = 255.0 * (x + 1.0) / 2.0

    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68

    return x

def flownet_preprocess(img_pair):
    # (-1, 1) to (0, 1)
    return img_pair / 2.0 + 0.5


class VGGLoss_CRN(nn.Module):
    def __init__(self, weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]):
        super(VGGLoss_CRN, self).__init__()
        self.vgg = VGG19_CRN()
        self.criterion = nn.L1Loss()
        self.weights = weights

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGG19_CRN(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19_CRN, self).__init__()
        self.vgg_model = vgg19(pretrained=True).features

        # replace max pooling with average pooling to eliminate grid effect
        mp_list = [4, 9, 18, 27, 36]
        for mp_idx in mp_list:
            self.vgg_model._modules[str(mp_idx)] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.extracted_layers = (lambda x: [str(i) for i in x])([2, 7, 12, 21, 30])
        # loss_layers are ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = []
        for name, module in self.vgg_model._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                feats.append(x)

        return feats


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvBlock(nn.Module):
    def __init__(self, n_repeats, c_in, c_out, kernel_size, pad):
        super(ConvBlock, self).__init__()
        self.conv_block = self.build_block(n_repeats, c_in, c_out, kernel_size, pad)

    def build_block(self, n_repeats, c_in, c_out, kernel_size, pad):
        conv_block = []
        for i in range(n_repeats):
            conv_block += [nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=pad),
                           LayerNorm(c_out),
                           nn.LeakyReLU()]
            c_in = c_out

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class CRN(nn.Module):
    def __init__(self, input_channel=6,fg=False):
        super(CRN, self).__init__()

        self.conv1_encoder = ConvBlock(2, c_in=input_channel, c_out=64, kernel_size=(3, 3), pad=1)
        self.conv2_encoder = ConvBlock(2, c_in=64, c_out=128, kernel_size=(3, 3), pad=1)
        self.conv3_encoder = ConvBlock(3, c_in=128, c_out=256, kernel_size=(3, 3), pad=1)
        self.conv4_encoder = ConvBlock(3, c_in=256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv5_encoder = ConvBlock(3, c_in=512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv6_encoder = ConvBlock(3, c_in=512, c_out=512, kernel_size=(3, 3), pad=1)

        self.conv6_decoder = ConvBlock(2, c_in=input_channel + 512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv5_decoder = ConvBlock(2, c_in=input_channel + 512 + 512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv4_decoder = ConvBlock(2, c_in=input_channel + 512 +512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv3_decoder = ConvBlock(2, c_in=input_channel + 512 + 256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv2_decoder = ConvBlock(2, c_in=input_channel + 512 + 128, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv1_decoder = ConvBlock(2, c_in=input_channel + 512 + 64, c_out=256, kernel_size=(3, 3), pad=1)

        self.decoder = ConvBlock(2, c_in=input_channel + 256, c_out=256, kernel_size=(3, 3), pad=1)
        self.out_conv = nn.Conv2d(256, 3, kernel_size=(1, 1))
        self.fg=fg
        if self.fg==True:
            self.fg_conv=nn.Conv2d(256, 1, kernel_size=(1, 1))

    def forward(self, label, sp):
        pool1 = F.avg_pool2d(self.conv1_encoder(label), (3, 3), stride=2, padding=1)
        pool2 = F.avg_pool2d(self.conv2_encoder(pool1), (3, 3), stride=2, padding=1)
        pool3 = F.avg_pool2d(self.conv3_encoder(pool2), (3, 3), stride=2, padding=1)
        pool4 = F.avg_pool2d(self.conv4_encoder(pool3), (3, 3), stride=2, padding=1)
        pool5 = F.avg_pool2d(self.conv5_encoder(pool4), (3, 3), stride=2, padding=1)
        pool6 = F.avg_pool2d(self.conv6_encoder(pool5), (3, 3), stride=2, padding=1)

        downsampled_6 = F.interpolate(label, sp // 64, mode='bilinear', align_corners=True)
        input_6 = torch.cat([downsampled_6, pool6], dim=1)
        net_6 = F.interpolate(self.conv6_decoder(input_6), sp // 32, mode='bilinear', align_corners=True)

        downsampled_5 = F.interpolate(label, sp // 32, mode='bilinear', align_corners=True)
        input_5 = torch.cat([downsampled_5, pool5, net_6], dim=1)
        net_5 = F.interpolate(self.conv5_decoder(input_5), sp // 16, mode='bilinear', align_corners=True)

        downsampled_4 = F.interpolate(label, sp // 16, mode='bilinear', align_corners=True)
        input_4 = torch.cat([downsampled_4, pool4, net_5], dim=1)
        net_4 = F.interpolate(self.conv4_decoder(input_4), sp // 8, mode='bilinear', align_corners=True)

        downsampled_3 = F.interpolate(label, sp // 8, mode='bilinear', align_corners=True)
        input_3 = torch.cat([downsampled_3, pool3, net_4], dim=1)
        net_3 = F.interpolate(self.conv3_decoder(input_3), sp // 4, mode='bilinear', align_corners=True)

        downsampled_2 = F.interpolate(label, sp // 4, mode='bilinear', align_corners=True)
        input_2 = torch.cat([downsampled_2, pool2, net_3], dim=1)
        net_2 = F.interpolate(self.conv2_decoder(input_2), sp // 2, mode='bilinear', align_corners=True)

        downsampled_1 = F.interpolate(label, sp // 2, mode='bilinear', align_corners=True)
        input_1 = torch.cat([downsampled_1, pool1, net_2], dim=1)
        net_1 = F.interpolate(self.conv1_decoder(input_1), sp, mode='bilinear', align_corners=True)

        input = torch.cat([label, net_1], dim=1)
        net = self.decoder(input)

        net_final = self.out_conv(net)

        if self.fg==True:
            fg_mask=F.sigmoid(self.fg_conv(net))
            return net_final,fg_mask

        return net_final
        
class CRN_small(nn.Module):
    def __init__(self, input_channel=6,fg=False):
        super(CRN_small, self).__init__()

        self.conv1_encoder = ConvBlock(2, c_in=input_channel, c_out=64, kernel_size=(3, 3), pad=1)
        self.conv2_encoder = ConvBlock(2, c_in=64, c_out=128, kernel_size=(3, 3), pad=1)
        self.conv3_encoder = ConvBlock(2, c_in=128, c_out=256, kernel_size=(3, 3), pad=1)
        self.conv4_encoder = ConvBlock(2, c_in=256, c_out=256, kernel_size=(3, 3), pad=1)
        self.conv5_encoder = ConvBlock(2, c_in=256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv6_encoder = ConvBlock(2, c_in=512, c_out=512, kernel_size=(3, 3), pad=1)

        self.conv6_decoder = ConvBlock(2, c_in=input_channel + 512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv5_decoder = ConvBlock(2, c_in=input_channel + 512 + 512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv4_decoder = ConvBlock(2, c_in=input_channel + 512 + 256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv3_decoder = ConvBlock(2, c_in=input_channel + 512 + 256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv2_decoder = ConvBlock(2, c_in=input_channel + 512 + 128, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv1_decoder = ConvBlock(2, c_in=input_channel + 512 + 64, c_out=256, kernel_size=(3, 3), pad=1)

        self.decoder = ConvBlock(2, c_in=input_channel + 256, c_out=256, kernel_size=(3, 3), pad=1)
        self.out_conv = nn.Conv2d(256, 3, kernel_size=(1, 1))
        self.fg=fg
        if self.fg==True:
            self.fg_conv=nn.Conv2d(256, 1, kernel_size=(1, 1))

    def forward(self, label, sp):
        pool1 = F.avg_pool2d(self.conv1_encoder(label), (3, 3), stride=2, padding=1)
        pool2 = F.avg_pool2d(self.conv2_encoder(pool1), (3, 3), stride=2, padding=1)
        pool3 = F.avg_pool2d(self.conv3_encoder(pool2), (3, 3), stride=2, padding=1)
        pool4 = F.avg_pool2d(self.conv4_encoder(pool3), (3, 3), stride=2, padding=1)
        pool5 = F.avg_pool2d(self.conv5_encoder(pool4), (3, 3), stride=2, padding=1)
        pool6 = F.avg_pool2d(self.conv6_encoder(pool5), (3, 3), stride=2, padding=1)

        downsampled_6 = F.interpolate(label, sp // 64, mode='bilinear', align_corners=True)
        input_6 = torch.cat([downsampled_6, pool6], dim=1)
        net_6 = F.interpolate(self.conv6_decoder(input_6), sp // 32, mode='bilinear', align_corners=True)

        downsampled_5 = F.interpolate(label, sp // 32, mode='bilinear', align_corners=True)
        input_5 = torch.cat([downsampled_5, pool5, net_6], dim=1)
        net_5 = F.interpolate(self.conv5_decoder(input_5), sp // 16, mode='bilinear', align_corners=True)

        downsampled_4 = F.interpolate(label, sp // 16, mode='bilinear', align_corners=True)
        input_4 = torch.cat([downsampled_4, pool4, net_5], dim=1)
        net_4 = F.interpolate(self.conv4_decoder(input_4), sp // 8, mode='bilinear', align_corners=True)

        downsampled_3 = F.interpolate(label, sp // 8, mode='bilinear', align_corners=True)
        input_3 = torch.cat([downsampled_3, pool3, net_4], dim=1)
        net_3 = F.interpolate(self.conv3_decoder(input_3), sp // 4, mode='bilinear', align_corners=True)

        downsampled_2 = F.interpolate(label, sp // 4, mode='bilinear', align_corners=True)
        input_2 = torch.cat([downsampled_2, pool2, net_3], dim=1)
        net_2 = F.interpolate(self.conv2_decoder(input_2), sp // 2, mode='bilinear', align_corners=True)

        downsampled_1 = F.interpolate(label, sp // 2, mode='bilinear', align_corners=True)
        input_1 = torch.cat([downsampled_1, pool1, net_2], dim=1)
        net_1 = F.interpolate(self.conv1_decoder(input_1), sp, mode='bilinear', align_corners=True)

        input = torch.cat([label, net_1], dim=1)
        net = self.decoder(input)

        net_final = self.out_conv(net)

        if self.fg==True:
            fg_mask=F.sigmoid(self.fg_conv(net))
            return net_final,fg_mask

        return net_final
        
class CRN_smaller(nn.Module):
    def __init__(self, input_channel=6,fg=False):
        super(CRN_smaller, self).__init__()

        self.conv1_encoder = ConvBlock(2, c_in=input_channel, c_out=64, kernel_size=(3, 3), pad=1)
        self.conv2_encoder = ConvBlock(2, c_in=64, c_out=128, kernel_size=(3, 3), pad=1)
        self.conv3_encoder = ConvBlock(2, c_in=128, c_out=128, kernel_size=(3, 3), pad=1)
        self.conv4_encoder = ConvBlock(2, c_in=128, c_out=256, kernel_size=(3, 3), pad=1)
        self.conv5_encoder = ConvBlock(2, c_in=256, c_out=256, kernel_size=(3, 3), pad=1)
        self.conv6_encoder = ConvBlock(2, c_in=256, c_out=512, kernel_size=(3, 3), pad=1)

        self.conv6_decoder = ConvBlock(2, c_in=input_channel + 512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv5_decoder = ConvBlock(2, c_in=input_channel + 512 + 256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv4_decoder = ConvBlock(2, c_in=input_channel + 512 + 256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv3_decoder = ConvBlock(2, c_in=input_channel + 512 + 128, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv2_decoder = ConvBlock(2, c_in=input_channel + 512 + 128, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv1_decoder = ConvBlock(2, c_in=input_channel + 512 + 64, c_out=256, kernel_size=(3, 3), pad=1)

        self.decoder = ConvBlock(2, c_in=input_channel + 256, c_out=256, kernel_size=(3, 3), pad=1)
        self.out_conv = nn.Conv2d(256, 3, kernel_size=(1, 1))
        self.fg=fg
        if self.fg==True:
            self.fg_conv=nn.Conv2d(256, 1, kernel_size=(1, 1))

    def forward(self, label, sp):
        pool1 = F.avg_pool2d(self.conv1_encoder(label), (3, 3), stride=2, padding=1)
        pool2 = F.avg_pool2d(self.conv2_encoder(pool1), (3, 3), stride=2, padding=1)
        pool3 = F.avg_pool2d(self.conv3_encoder(pool2), (3, 3), stride=2, padding=1)
        pool4 = F.avg_pool2d(self.conv4_encoder(pool3), (3, 3), stride=2, padding=1)
        pool5 = F.avg_pool2d(self.conv5_encoder(pool4), (3, 3), stride=2, padding=1)
        pool6 = F.avg_pool2d(self.conv6_encoder(pool5), (3, 3), stride=2, padding=1)

        downsampled_6 = F.interpolate(label, sp // 64, mode='bilinear', align_corners=True)
        input_6 = torch.cat([downsampled_6, pool6], dim=1)
        net_6 = F.interpolate(self.conv6_decoder(input_6), sp // 32, mode='bilinear', align_corners=True)

        downsampled_5 = F.interpolate(label, sp // 32, mode='bilinear', align_corners=True)
        input_5 = torch.cat([downsampled_5, pool5, net_6], dim=1)
        net_5 = F.interpolate(self.conv5_decoder(input_5), sp // 16, mode='bilinear', align_corners=True)

        downsampled_4 = F.interpolate(label, sp // 16, mode='bilinear', align_corners=True)
        input_4 = torch.cat([downsampled_4, pool4, net_5], dim=1)
        net_4 = F.interpolate(self.conv4_decoder(input_4), sp // 8, mode='bilinear', align_corners=True)

        downsampled_3 = F.interpolate(label, sp // 8, mode='bilinear', align_corners=True)
        input_3 = torch.cat([downsampled_3, pool3, net_4], dim=1)
        net_3 = F.interpolate(self.conv3_decoder(input_3), sp // 4, mode='bilinear', align_corners=True)

        downsampled_2 = F.interpolate(label, sp // 4, mode='bilinear', align_corners=True)
        input_2 = torch.cat([downsampled_2, pool2, net_3], dim=1)
        net_2 = F.interpolate(self.conv2_decoder(input_2), sp // 2, mode='bilinear', align_corners=True)

        downsampled_1 = F.interpolate(label, sp // 2, mode='bilinear', align_corners=True)
        input_1 = torch.cat([downsampled_1, pool1, net_2], dim=1)
        net_1 = F.interpolate(self.conv1_decoder(input_1), sp, mode='bilinear', align_corners=True)

        input = torch.cat([label, net_1], dim=1)
        net = self.decoder(input)

        net_final = self.out_conv(net)

        if self.fg==True:
            fg_mask=F.sigmoid(self.fg_conv(net))
            return net_final,fg_mask

        return net_final
        
class AutoEncoder(nn.Module):
    def __init__(self,input_channel=3):
        super(AutoEncoder,self).__init__()
        
        self.conv1_encoder = ConvBlock(2, c_in=input_channel, c_out=16, kernel_size=(3, 3), pad=1)
        self.conv2_encoder = ConvBlock(2, c_in=16, c_out=32, kernel_size=(3, 3), pad=1)
        self.conv3_encoder = ConvBlock(3, c_in=32, c_out=64, kernel_size=(3, 3), pad=1)
        self.conv4_encoder = ConvBlock(3, c_in=64, c_out=64, kernel_size=(3, 3), pad=1)
        self.conv5_encoder = ConvBlock(3, c_in=64, c_out=128, kernel_size=(3, 3), pad=1)
        self.conv6_encoder = ConvBlock(3, c_in=128, c_out=128, kernel_size=(3, 3), pad=1)
        
    def forward(self,src_img):
        pool1=F.max_pool2d(self.conv1_encoder(src_img), (3, 3), stride=2, padding=1)
        pool2 = F.max_pool2d(self.conv2_encoder(pool1), (3, 3), stride=2, padding=1)
        pool3 = F.max_pool2d(self.conv3_encoder(pool2), (3, 3), stride=2, padding=1)
        pool4 = F.max_pool2d(self.conv4_encoder(pool3), (3, 3), stride=2, padding=1)
        pool5 = F.max_pool2d(self.conv5_encoder(pool4), (3, 3), stride=2, padding=1)
        pool6 = F.max_pool2d(self.conv6_encoder(pool5), (3, 3), stride=2, padding=1)
        
        return pool6
        
class CRN_Auto(nn.Module):
    def __init__(self, input_channel=6):
        super(CRN_Auto, self).__init__()

        self.conv1_encoder = ConvBlock(2, c_in=input_channel, c_out=64, kernel_size=(3, 3), pad=1)
        self.conv2_encoder = ConvBlock(2, c_in=64, c_out=128, kernel_size=(3, 3), pad=1)
        self.conv3_encoder = ConvBlock(3, c_in=128, c_out=256, kernel_size=(3, 3), pad=1)
        self.conv4_encoder = ConvBlock(3, c_in=256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv5_encoder = ConvBlock(3, c_in=512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv6_encoder = ConvBlock(3, c_in=512, c_out=512, kernel_size=(3, 3), pad=1)

        self.conv6_decoder = ConvBlock(2, c_in=input_channel + 512+128, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv5_decoder = ConvBlock(2, c_in=input_channel + 512 + 512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv4_decoder = ConvBlock(2, c_in=input_channel + 512 +512, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv3_decoder = ConvBlock(2, c_in=input_channel + 512 + 256, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv2_decoder = ConvBlock(2, c_in=input_channel + 512 + 128, c_out=512, kernel_size=(3, 3), pad=1)
        self.conv1_decoder = ConvBlock(2, c_in=input_channel + 512 + 64, c_out=256, kernel_size=(3, 3), pad=1)
        
        self.AutoEncoder=AutoEncoder(input_channel=3)

        self.decoder = ConvBlock(2, c_in=input_channel + 256, c_out=256, kernel_size=(3, 3), pad=1)
        self.out_conv = nn.Conv2d(256, 3, kernel_size=(1, 1))

        self.criterion = nn.L1Loss()
        self.perceptual_loss = VGGLoss_CRN(weights=[1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5])

    def forward(self, label, sp, real,src_img):
        embed=self.AutoEncoder(src_img)
        pool1 = F.avg_pool2d(self.conv1_encoder(label), (3, 3), stride=2, padding=1)
        pool2 = F.avg_pool2d(self.conv2_encoder(pool1), (3, 3), stride=2, padding=1)
        pool3 = F.avg_pool2d(self.conv3_encoder(pool2), (3, 3), stride=2, padding=1)
        pool4 = F.avg_pool2d(self.conv4_encoder(pool3), (3, 3), stride=2, padding=1)
        pool5 = F.avg_pool2d(self.conv5_encoder(pool4), (3, 3), stride=2, padding=1)
        pool6 = F.avg_pool2d(self.conv6_encoder(pool5), (3, 3), stride=2, padding=1)

        downsampled_6 = F.interpolate(label, sp // 64, mode='bilinear', align_corners=True)
        input_6 = torch.cat([downsampled_6, pool6,embed], dim=1)
        net_6 = F.interpolate(self.conv6_decoder(input_6), sp // 32, mode='bilinear', align_corners=True)

        downsampled_5 = F.interpolate(label, sp // 32, mode='bilinear', align_corners=True)
        input_5 = torch.cat([downsampled_5, pool5, net_6], dim=1)
        net_5 = F.interpolate(self.conv5_decoder(input_5), sp // 16, mode='bilinear', align_corners=True)

        downsampled_4 = F.interpolate(label, sp // 16, mode='bilinear', align_corners=True)
        input_4 = torch.cat([downsampled_4, pool4, net_5], dim=1)
        net_4 = F.interpolate(self.conv4_decoder(input_4), sp // 8, mode='bilinear', align_corners=True)

        downsampled_3 = F.interpolate(label, sp // 8, mode='bilinear', align_corners=True)
        input_3 = torch.cat([downsampled_3, pool3, net_4], dim=1)
        net_3 = F.interpolate(self.conv3_decoder(input_3), sp // 4, mode='bilinear', align_corners=True)

        downsampled_2 = F.interpolate(label, sp // 4, mode='bilinear', align_corners=True)
        input_2 = torch.cat([downsampled_2, pool2, net_3], dim=1)
        net_2 = F.interpolate(self.conv2_decoder(input_2), sp // 2, mode='bilinear', align_corners=True)

        downsampled_1 = F.interpolate(label, sp // 2, mode='bilinear', align_corners=True)
        input_1 = torch.cat([downsampled_1, pool1, net_2], dim=1)
        net_1 = F.interpolate(self.conv1_decoder(input_1), sp, mode='bilinear', align_corners=True)

        input = torch.cat([label, net_1], dim=1)
        net = self.decoder(input)

        net_final = self.out_conv(net)

        # loss = 0.
        # calculate L1 loss
        loss = self.criterion(vgg_preprocess(net_final), vgg_preprocess(real))

        # calculate perceptual loss
        loss += self.perceptual_loss(vgg_preprocess(net_final), vgg_preprocess(real))

        return loss, net_final

class SpatioTempoCRN(nn.Module):
    def __init__(self, opt, input_channel=6, ngf=512):
        super(SpatioTempoCRN, self).__init__()
        self.isTrain = opt['isTrain']

        self.conv1_encoder = ConvBlock(2, c_in=input_channel, c_out=64, kernel_size=(3, 3), pad=1)
        self.conv2_encoder = ConvBlock(2, c_in=64, c_out=128, kernel_size=(3, 3), pad=1)
        self.conv3_encoder = ConvBlock(3, c_in=128, c_out=256, kernel_size=(3, 3), pad=1)
        self.conv4_encoder = ConvBlock(3, c_in=256, c_out=ngf, kernel_size=(3, 3), pad=1)
        self.conv5_encoder = ConvBlock(3, c_in=ngf, c_out=ngf, kernel_size=(3, 3), pad=1)
        self.conv6_encoder = ConvBlock(3, c_in=ngf, c_out=ngf, kernel_size=(3, 3), pad=1)

        self.conv6_decoder = ConvBlock(2, c_in=input_channel + ngf * 2, c_out=ngf, kernel_size=(3, 3), pad=1)
        self.conv5_decoder = ConvBlock(2, c_in=input_channel + ngf + ngf * 2, c_out=ngf, kernel_size=(3, 3), pad=1)
        self.conv4_decoder = ConvBlock(2, c_in=input_channel + ngf + ngf * 2, c_out=ngf, kernel_size=(3, 3), pad=1)
        self.conv3_decoder = ConvBlock(2, c_in=input_channel + ngf + 256 * 2, c_out=ngf, kernel_size=(3, 3), pad=1)
        self.conv2_decoder = ConvBlock(2, c_in=input_channel + ngf + 128 * 2, c_out=ngf, kernel_size=(3, 3), pad=1)
        self.conv1_decoder = ConvBlock(2, c_in=input_channel + ngf + 64 * 2, c_out=256, kernel_size=(3, 3), pad=1)

        self.decoder = ConvBlock(2, c_in=input_channel + 256, c_out=256, kernel_size=(3, 3), pad=1)
        self.out_conv = nn.Conv2d(256, 3, kernel_size=(1, 1))

        self.criterion = nn.L1Loss()
        self.perceptual_loss = VGGLoss_CRN(weights=[1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5])

        self.flownet = FlowNetSD(args=[], batchNorm=False, requires_grad=False)
        self.flownet.load_state_dict(torch.load(opt['flownet_path'])['state_dict'])

    def forward(self, label, prev_label, sp, real, prev_real, iuv_pair, grid_list):
        N, C, H, W = label.size()
        if iuv_pair is None:  # only in test mode
            flow = torch.zeros((N, 2, H, W)).cuda()
        else:
            flow = self.flownet(flownet_preprocess(iuv_pair))[0]

        pool1 = F.avg_pool2d(self.conv1_encoder(label), (3, 3), stride=2, padding=1)
        pool2 = F.avg_pool2d(self.conv2_encoder(pool1), (3, 3), stride=2, padding=1)
        pool3 = F.avg_pool2d(self.conv3_encoder(pool2), (3, 3), stride=2, padding=1)
        pool4 = F.avg_pool2d(self.conv4_encoder(pool3), (3, 3), stride=2, padding=1)
        pool5 = F.avg_pool2d(self.conv5_encoder(pool4), (3, 3), stride=2, padding=1)
        pool6 = F.avg_pool2d(self.conv6_encoder(pool5), (3, 3), stride=2, padding=1)

        prev_pool1 = F.avg_pool2d(self.conv1_encoder(prev_label), (3, 3), stride=2, padding=1)
        prev_pool2 = F.avg_pool2d(self.conv2_encoder(prev_pool1), (3, 3), stride=2, padding=1)
        prev_pool3 = F.avg_pool2d(self.conv3_encoder(prev_pool2), (3, 3), stride=2, padding=1)
        prev_pool4 = F.avg_pool2d(self.conv4_encoder(prev_pool3), (3, 3), stride=2, padding=1)
        prev_pool5 = F.avg_pool2d(self.conv5_encoder(prev_pool4), (3, 3), stride=2, padding=1)
        prev_pool6 = F.avg_pool2d(self.conv6_encoder(prev_pool5), (3, 3), stride=2, padding=1)

        # 6
        downsampled_6 = F.interpolate(label, sp // 64, mode='bilinear', align_corners=True)
        prev_downsampled_6 = F.interpolate(prev_label, sp // 64, mode='bilinear', align_corners=True)

        downsampled_6_flow = F.interpolate(flow, sp // 64, mode='nearest')
        # N, C, H, W = prev_pool6.size()
        # grid = get_grid(N, H, W).cuda()
        grid = grid_list[6 - 1]
        grid.requires_grad = False
 
        warped_prev_pool6 = F.grid_sample(input=prev_pool6, grid=(grid + downsampled_6_flow).permute(0, 2, 3, 1),
                                          padding_mode='border')
        warped_pool6 = F.grid_sample(input=pool6, grid=(grid - downsampled_6_flow).permute(0, 2, 3, 1),
                                     padding_mode='border')

        input_6 = torch.cat([downsampled_6, pool6, warped_prev_pool6], dim=1)
        prev_input_6 = torch.cat([prev_downsampled_6, prev_pool6, warped_pool6], dim=1)

        net_6 = F.interpolate(self.conv6_decoder(input_6), sp // 32, mode='bilinear', align_corners=True)
        prev_net_6 = F.interpolate(self.conv6_decoder(prev_input_6), sp // 32, mode='bilinear', align_corners=True)

        # 5
        downsampled_5 = F.interpolate(label, sp // 32, mode='bilinear', align_corners=True)
        prev_downsampled_5 = F.interpolate(prev_label, sp // 32, mode='bilinear', align_corners=True)

        downsampled_5_flow = F.interpolate(flow, sp // 32, mode='nearest')
        # N, C, H, W = prev_pool5.size()
        # grid = get_grid(N, H, W).cuda()
        grid = grid_list[5 - 1]
        grid.requires_grad = False
        warped_prev_pool5 = F.grid_sample(input=prev_pool5, grid=(grid + downsampled_5_flow).permute(0, 2, 3, 1),
                                          padding_mode='border')
        warped_pool5 = F.grid_sample(input=pool5, grid=(grid - downsampled_5_flow).permute(0, 2, 3, 1),
                                     padding_mode='border')

        input_5 = torch.cat([downsampled_5, pool5, net_6, warped_prev_pool5], dim=1)
        prev_input_5 = torch.cat([prev_downsampled_5, prev_pool5, prev_net_6, warped_pool5], dim=1)

        net_5 = F.interpolate(self.conv5_decoder(input_5), sp // 16, mode='bilinear', align_corners=True)
        prev_net_5 = F.interpolate(self.conv5_decoder(prev_input_5), sp // 16, mode='bilinear', align_corners=True)

        # 4
        downsampled_4 = F.interpolate(label, sp // 16, mode='bilinear', align_corners=True)
        prev_downsampled_4 = F.interpolate(prev_label, sp // 16, mode='bilinear', align_corners=True)

        downsampled_4_flow = F.interpolate(flow, sp // 16, mode='nearest')
        # N, C, H, W = prev_pool4.size()
        # grid = get_grid(N, H, W).cuda()
        grid = grid_list[4 - 1]
        grid.requires_grad = False
        warped_prev_pool4 = F.grid_sample(input=prev_pool4, grid=(grid + downsampled_4_flow).permute(0, 2, 3, 1),
                                          padding_mode='border')
        warped_pool4 = F.grid_sample(input=pool4, grid=(grid - downsampled_4_flow).permute(0, 2, 3, 1),
                                     padding_mode='border')

        input_4 = torch.cat([downsampled_4, pool4, net_5, warped_prev_pool4], dim=1)
        prev_input_4 = torch.cat([prev_downsampled_4, prev_pool4, prev_net_5, warped_pool4], dim=1)

        net_4 = F.interpolate(self.conv4_decoder(input_4), sp // 8, mode='bilinear', align_corners=True)
        prev_net_4 = F.interpolate(self.conv4_decoder(prev_input_4), sp // 8, mode='bilinear', align_corners=True)

        # 3
        downsampled_3 = F.interpolate(label, sp // 8, mode='bilinear', align_corners=True)
        prev_downsampled_3 = F.interpolate(prev_label, sp // 8, mode='bilinear', align_corners=True)

        downsampled_3_flow = F.interpolate(flow, sp // 8, mode='nearest')
        # N, C, H, W = prev_pool3.size()
        # grid = get_grid(N, H, W).cuda()
        grid = grid_list[3 - 1]
        grid.requires_grad = False
        warped_prev_pool3 = F.grid_sample(input=prev_pool3, grid=(grid + downsampled_3_flow).permute(0, 2, 3, 1),
                                          padding_mode='border')
        warped_pool3 = F.grid_sample(input=pool3, grid=(grid - downsampled_3_flow).permute(0, 2, 3, 1),
                                     padding_mode='border')

        input_3 = torch.cat([downsampled_3, pool3, net_4, warped_prev_pool3], dim=1)
        prev_input_3 = torch.cat([prev_downsampled_3, prev_pool3, prev_net_4, warped_pool3], dim=1)

        net_3 = F.interpolate(self.conv3_decoder(input_3), sp // 4, mode='bilinear', align_corners=True)
        prev_net_3 = F.interpolate(self.conv3_decoder(prev_input_3), sp // 4, mode='bilinear', align_corners=True)

        # 2
        downsampled_2 = F.interpolate(label, sp // 4, mode='bilinear', align_corners=True)
        prev_downsampled_2 = F.interpolate(prev_label, sp // 4, mode='bilinear', align_corners=True)

        downsampled_2_flow = F.interpolate(flow, sp // 4, mode='nearest')
        # N, C, H, W = prev_pool2.size()
        # grid = get_grid(N, H, W).cuda()
        grid = grid_list[2 - 1]
        grid.requires_grad = False
        warped_prev_pool2 = F.grid_sample(input=prev_pool2, grid=(grid + downsampled_2_flow).permute(0, 2, 3, 1),
                                          padding_mode='border')
        warped_pool2 = F.grid_sample(input=pool2, grid=(grid - downsampled_2_flow).permute(0, 2, 3, 1),
                                     padding_mode='border')

        input_2 = torch.cat([downsampled_2, pool2, net_3, warped_prev_pool2], dim=1)
        prev_input_2 = torch.cat([prev_downsampled_2, prev_pool2, prev_net_3, warped_pool2], dim=1)

        net_2 = F.interpolate(self.conv2_decoder(input_2), sp // 2, mode='bilinear', align_corners=True)
        prev_net_2 = F.interpolate(self.conv2_decoder(prev_input_2), sp // 2, mode='bilinear', align_corners=True)

        # 1
        downsampled_1 = F.interpolate(label, sp // 2, mode='bilinear', align_corners=True)
        prev_downsampled_1 = F.interpolate(prev_label, sp // 2, mode='bilinear', align_corners=True)

        downsampled_1_flow = F.interpolate(flow, sp // 2, mode='nearest')
        # N, C, H, W = prev_pool1.size()
        # grid = get_grid(N, H, W).cuda()
        grid = grid_list[1 - 1]
        grid.requires_grad = False
        warped_prev_pool1 = F.grid_sample(input=prev_pool1, grid=(grid + downsampled_1_flow).permute(0, 2, 3, 1),
                                          padding_mode='border')
        warped_pool1 = F.grid_sample(input=pool1, grid=(grid - downsampled_1_flow).permute(0, 2, 3, 1),
                                     padding_mode='border')

        input_1 = torch.cat([downsampled_1, pool1, net_2, warped_prev_pool1], dim=1)
        prev_input_1 = torch.cat([prev_downsampled_1, prev_pool1, prev_net_2, warped_pool1], dim=1)

        net_1 = F.interpolate(self.conv1_decoder(input_1), sp, mode='bilinear', align_corners=True)
        prev_net_1 = F.interpolate(self.conv1_decoder(prev_input_1), sp, mode='bilinear', align_corners=True)

        input = torch.cat([label, net_1], dim=1)
        prev_input = torch.cat([prev_label, prev_net_1], dim=1)

        net = self.decoder(input)
        prev_net = self.decoder(prev_input)

        net_final = self.out_conv(net)
        prev_net_final = self.out_conv(prev_net)

        if self.isTrain:
            # loss = 0.
            # calculate L1 loss
            loss = self.criterion(vgg_preprocess(net_final), vgg_preprocess(real)) + self.criterion(
                vgg_preprocess(prev_net_final), vgg_preprocess(prev_real))

            # calculate perceptual loss
            loss += self.perceptual_loss(vgg_preprocess(net_final), vgg_preprocess(real))
            loss += self.perceptual_loss(vgg_preprocess(prev_net_final), vgg_preprocess(prev_real))

            # calculate flow loss
            real_pair = torch.cat([prev_real, real], dim=1)
            fake_pair = torch.cat([prev_net_final, net_final], dim=1)
            flow_loss = self.criterion(self.flownet(flownet_preprocess(fake_pair))[0],
                                       self.flownet(flownet_preprocess(real_pair))[0])

            return loss, flow_loss, net_final, prev_net_final
        else:
            return net_final, prev_net_final

