import torch
from torch import nn
import torch.nn.functional as F


class CompositeWeightUnet(nn.Module):
    def __init__(self, input_nc, ngf, n_downsampling, n_blocks, norm_layer=nn.BatchNorm2d, act=nn.ReLU,
                 padding_type='reflect', use_deconv=False, use_tgt_dp=False):
        super(CompositeWeightUnet, self).__init__()
        ### flow and image generation
        ### downsample
        input_nc = input_nc + 3 * use_tgt_dp
        model_down_img = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                          act]
        for i in range(n_downsampling):
            mult = 2 ** i
            model_down_img += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                               norm_layer(ngf * mult * 2), act]

        mult = 2 ** n_downsampling
        for i in range(n_blocks - n_blocks // 2):
            model_down_img += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=act, norm_layer=norm_layer)]

        ### resnet blocks
        model_res_img = []
        for i in range(n_blocks // 2):
            model_res_img += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=act, norm_layer=norm_layer)]

        ### upsample
        model_up_img = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if use_deconv:
                model_up_img += [
                    nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    norm_layer(ngf * mult // 2), act]
            else:
                model_up_img += [
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    nn.Conv2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=1, padding=1),
                    norm_layer(ngf * mult // 2), act]

        model_final_w = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Sigmoid()]

        self.model_down_img = nn.Sequential(*model_down_img)
        self.model_res_img = nn.Sequential(*model_res_img)
        self.model_up_img = nn.Sequential(*model_up_img)
        self.model_final_w = nn.Sequential(*model_final_w)

    def forward(self, input):
        downsample = self.model_down_img(input)
        img_feat = self.model_up_img(self.model_res_img(downsample))
        weight = self.model_final_w(img_feat)

        return weight


class Propagation3DFlowNet(nn.Module):
    def __init__(self, input_nc, ngf, n_downsampling, n_blocks, norm_type='batch', act_type='relu',
                 padding_type='reflect', use_deconv=True, use_tgt_dp=False):
        super(Propagation3DFlowNet, self).__init__()
        norm_dict = {
            'batch': nn.BatchNorm2d,
            'instance': nn.InstanceNorm2d
        }
        assert norm_type in norm_dict.keys(), 'Unrecongnized norm type: %s' % norm_type
        norm_layer = norm_dict[norm_type]

        activations = {
            'lrelu': nn.LeakyReLU(0.2, inplace=True),
            'relu': nn.ReLU(True),
            'linear': None,
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        assert act_type in activations.keys(), 'Unrecongnized activation type: %s' % act_type
        act = activations[act_type]

        self.composite_unet = CompositeWeightUnet(input_nc, ngf, n_downsampling, n_blocks, norm_layer, act,
                                                  padding_type, use_deconv, use_tgt_dp)

        self.use_tgt_dp = use_tgt_dp

    def forward(self, x):
        fake_tgt, tsf_image,tgt_IUV,use_IUV=  x['fake_tgt'], x['tsf_image'],x['tgt_IUV'],x['use_IUV']
        use_mask, tgt_smpl_mask = x['use_mask'], x['tgt_smpl_mask']

        tsf_image = tsf_image * tgt_smpl_mask if use_mask else tsf_image
        if not use_IUV:
            cated_input = torch.cat([tsf_image, fake_tgt], dim=1)
        else:
            cated_input = torch.cat([tsf_image, fake_tgt, tgt_IUV],dim=1)
        weight = self.composite_unet(cated_input)

        pred = fake_tgt * weight + tsf_image * (1 - weight)
        return {'pred_target': pred, 'weight': weight}


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
