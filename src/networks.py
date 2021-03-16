import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.autograd import Variable
from torch.nn import init
import math
from torchvision import models
from src.convLSTM import ConvLSTM,convGRU,ModconvGRU
from src.crn_model import CRN_smaller
import functools
import os

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

def texture_warp_pytorch(tex_parts, IUV):
    U = IUV[:,:,1]
    V = IUV[:,:,2]
    #
#     R_im = torch.zeros(U.size())
#     G_im = torch.zeros(U.size())
#     B_im = torch.zeros(U.size())
    generated_image = torch.zeros(IUV.size()).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    ###
    for PartInd in range(1,25):    ## Set to xrange(1,23) to ignore the face part.
#         tex = TextureIm[PartInd-1,:,:,:].squeeze() # get texture for each part.
        tex = tex_parts[PartInd - 1] # get texture for each part.
        #####
#         R = tex[:,:,0]
#         G = tex[:,:,1]
#         B = tex[:,:,2]
        ###############
#         x,y = torch.where(IUV[:,:,0]==PartInd, )
#         u_current_points = U[x,y]   #  Pixels that belong to this specific part.
#         v_current_points = V[x,y]
        u_current_points = torch.where(IUV[:,:,0]==PartInd, U.float().cuda(), torch.zeros(U.size()).cuda())   #  Pixels that belong to this specific part.
        v_current_points = torch.where(IUV[:,:,0]==PartInd, V.float().cuda(), torch.zeros(V.size()).cuda()) 
        
        x = ((255 - v_current_points) / 255. - 0.5) *2 # normalize to -1, 1
        y = (u_current_points / 255. - 0.5) *2
        grid = torch.cat([x.unsqueeze(2), y.unsqueeze(2)], dim=2).unsqueeze(0).cuda() # 1, H, W, 2
        tex_image = tex.unsqueeze(0).float().cuda() # 1, 3, H, W 
        
        sampled_patch = torch.nn.functional.grid_sample(tex_image, grid, mode='bilinear').cuda()
        generated_image = torch.where(IUV[:,:,0]==PartInd, sampled_patch.cuda(), generated_image.cuda())
        

    return generated_image.squeeze()
    
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
        
def vgg_preprocess(x):
    x = 255.0 * (x + 1.0) / 2.0

    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68

    return x
        
class VGG_l1_loss(nn.Module):
    def __init__(self):
        super(VGG_l1_loss,self).__init__()
        self.vgg_loss=VGGLoss_CRN(weights=[1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5])
        self.l1_loss=nn.L1Loss()
    def forward(self,x,y):
        loss=self.vgg_loss(vgg_preprocess(x),vgg_preprocess(y))+self.l1_loss(vgg_preprocess(x),vgg_preprocess(y))
        return loss

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class ImageDiscriminator(nn.Module):
    def __init__(self, ndf,input_channel=3):
        super(ImageDiscriminator, self).__init__()

        # use dcgan conventions
        self.main = nn.Sequential(
            # output size, num channels
            # 128, ndf
            nn.Conv2d(input_channel, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 64, ndf * 2
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32, ndf * 2
            nn.Conv2d(ndf * 2, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16, ndf * 4
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8, ndf * 4
            nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4, ndf * 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # # 2, ndf * 8
            # nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # # 1, ndf * 4
            # nn.Conv2d(ndf * 8, ndf * 4, 3, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # # 1, 1
            # nn.Conv2d(ndf * 4, 1, 3, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        # return output.view(-1, 1).squeeze(1)
        return output 
        
class FaceDiscriminator(nn.Module):
    def __init__(self, ndf,input_channel=3): #input channel is 6,64,64
        super(FaceDiscriminator, self).__init__()

        # use dcgan conventions
        self.main = nn.Sequential(
            # output size, num channels
            # 32, ndf
            nn.Conv2d(input_channel, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 16, ndf * 2
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 8, ndf * 2
            nn.Conv2d(ndf * 2, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 4, ndf * 4
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(ndf * 4 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        # return output.view(-1, 1).squeeze(1)
        return output 
        
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
        
class encoder(nn.Module):
    def __init__(self,input_channel=3): #input texture size is 200*200*3
        super(encoder, self).__init__()
        
        self.conv1_encoder = ConvBlock(1, c_in=3, c_out=16, kernel_size=(3, 3), pad=1)#input 100*100
        self.conv2_encoder = ConvBlock(1, c_in=16, c_out=32, kernel_size=(3, 3), pad=1)#input 50*50
        self.conv3_encoder = ConvBlock(1, c_in=32, c_out=32, kernel_size=(3, 3), pad=1)#input 50*50
        self.conv4_encoder = ConvBlock(1, c_in=32, c_out=64, kernel_size=(3, 3), pad=1)#input 25*25
        self.conv5_encoder = ConvBlock(1, c_in=64, c_out=64, kernel_size=(3, 3), pad=1)#input 12*12
        self.conv6_encoder = ConvBlock(1, c_in=64, c_out=128, kernel_size=(3, 3), pad=1)#input 6*6
        self.conv7_encoder = ConvBlock(1, c_in=128, c_out=256, kernel_size=(3, 3), pad=1)#input 3*3
        #output 1*1*64
    def forward(self,input_texture):
        pool1=F.max_pool2d(self.conv1_encoder(input_texture),kernel_size=(3, 3), stride=2, padding=1)#output 100*100
        pool2=F.max_pool2d(self.conv2_encoder(pool1),kernel_size=(3, 3), stride=2, padding=1)#output 50*50
        pool3=F.max_pool2d(self.conv3_encoder(pool2),kernel_size=(3, 3), stride=2, padding=1)#output 25*25
        pool4=F.max_pool2d(self.conv4_encoder(pool3),kernel_size=(3, 3), stride=2, padding=0)#output 
        pool5=F.max_pool2d(self.conv5_encoder(pool4),kernel_size=(3, 3), stride=2, padding=1)
        pool6=F.max_pool2d(self.conv6_encoder(pool5),kernel_size=(3, 3), stride=2, padding=1)
        pool7=F.max_pool2d(self.conv7_encoder(pool6),kernel_size=(3, 3), stride=2, padding=0)
        net_code=pool7.squeeze(2)
        net_code=net_code.squeeze(2)
        net_code=net_code.unsqueeze(1)
        #eventually output shape is batch_size,1,256]
        #print("net code's shape is",net_code.shape)
        return net_code

class decoder(nn.Module):
    def __init__(self,output_channel=3):
        super(decoder,self).__init__()#input is 512, 1*1*512

        self.deconv0_decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256, kernel_size=(3, 3),stride=2,padding=0),
            nn.LeakyReLU()
        )
        self.deconv1_decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128, kernel_size=(4, 4),stride=2,padding=1),
            nn.LeakyReLU()
        )#input 3*3
        self.deconv2_decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64, kernel_size=(4, 4),stride=2,padding=1),
            nn.LeakyReLU()
        )#input 6*6
        self.deconv3_decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32, kernel_size=(3, 3),stride=2,padding=0),
            nn.LeakyReLU()
        )#input 12*12
        self.deconv4_decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16, kernel_size=(4, 4),stride=2,padding=1),
            nn.LeakyReLU()
        )#input 25*25
        self.deconv5_decoder = nn.Sequential(
            nn.ConvTranspose2d(16,16, kernel_size=(4, 4),stride=2,padding=1),
            nn.LeakyReLU()
        )#input 50*50
        self.deconv6_decoder = nn.Sequential(
            nn.ConvTranspose2d(16,16, kernel_size=(4, 4),stride=2,padding=1),
            nn.LeakyReLU()
        )#input 100*100

        self.conv_out=nn.Sequential(
            nn.Conv2d(16,output_channel, kernel_size=(1, 1)),
            nn.Tanh()
        )

    def forward(self,latent_code):

        code=latent_code
        code=code.squeeze(1)
        code=code.unsqueeze(2)
        code=code.unsqueeze(2)#shape is batchsize 320 1 1
        up0=self.deconv0_decoder(code)
        up1=self.deconv1_decoder(up0)
        #print(up1.shape)
        up2=self.deconv2_decoder(up1)
        #print(up2.shape)
        up3=self.deconv3_decoder(up2)
        #print(up3.shape)
        up4=self.deconv4_decoder(up3)
        #print(up4.shape)
        up5=self.deconv5_decoder(up4)
        #print(up5.shape)
        up6=self.deconv6_decoder(up5)
        #print(up6.shape)

        net_out=self.conv_out(up6)

        return net_out
        

class max_fusion_module(nn.Module):
    def __init__(self):
          super(max_fusion_module,self).__init__()
          encoder_list=[]
          decoder_list=[]

          for i in range(24):
              encoder_list.append(encoder(input_channel=3))
              decoder_list.append(decoder(output_channel=3))
          self.encoder_list=nn.ModuleList(encoder_list)
          self.decoder_list=nn.ModuleList(decoder_list)

          self.fc=nn.Sequential(
              nn.Linear(6144,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
              nn.Linear(256,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
              nn.Linear(256,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
              nn.Linear(256,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
              nn.Linear(256,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
          )

          self.criterion=nn.L1Loss()
          
    def forward(self,bat_input_texture,tgt_texture,tgt_mask):
        inpaint_texture_list=[]
        global_code_list=[]
        project_code_list=[]
        for k in range(bat_input_texture.shape[1]):
            code_list=[]
            for i in range(4):
                for j in range(6):
                    body_part_texture=bat_input_texture[:,k,:,i*200:i*200+200,j*200:j*200+200]
                    encoder_index=i*6+j
                    encoder=self.encoder_list[encoder_index]
                    code_list.append(encoder(body_part_texture))
            global_code=torch.cat(code_list,dim=2)#global code is 6144
            global_code_list.append(global_code)
            #print("global_code's shape is",global_code.shape)
            project_code=self.fc(global_code) # project code is 256
            project_code_list.append(project_code.unsqueeze(1))
        
        
        bat_global_code=torch.cat(global_code_list,dim=1)
        bat_project_code=torch.cat(project_code_list,dim=1)
        
        fus_global_code=torch.max(bat_global_code,dim=1)[0]
        fus_project_code=torch.max(bat_project_code,dim=1)[0]
        
        #print(fus_project_code.shape)
        
        fus_part_code=fus_global_code.view(bat_global_code.shape[0],24,-1)
            
        for i in range(4):
            for j in range(6):
                decoder_index=i*6+j
                decoder=self.decoder_list[decoder_index]
                input_code=torch.cat([fus_project_code.squeeze(1),fus_part_code[:,decoder_index].squeeze(1)],dim=1)#input code 512
                inpaint_texture_list.append(decoder(input_code))

        for i in range(4):
            for j in range(6):
                index=i*6+j
                result_texture=inpaint_texture_list[index]
                for k in range(3):
                    target_texture=tgt_texture[:,k,:,i*200:i*200+200,j*200:j*200+200]
                    target_mask=tgt_mask[:,k,i*200:i*200+200,j*200:j*200+200]
                    loss_area=torch.cat([target_mask.unsqueeze(1),target_mask.unsqueeze(1),target_mask.unsqueeze(1)],dim=1)
                    if index==0 and k==0:
                        loss=self.criterion(result_texture*loss_area,target_texture*loss_area)
                    else:
                        loss=loss+self.criterion(result_texture*loss_area,target_texture*loss_area)
                
        return loss, inpaint_texture_list
        
class max_fusion_no_loss(nn.Module):
    def __init__(self):
          super(max_fusion_no_loss,self).__init__()
          encoder_list=[]
          decoder_list=[]

          for i in range(24):
              encoder_list.append(encoder(input_channel=3))
              decoder_list.append(decoder(output_channel=3))
          self.encoder_list=nn.ModuleList(encoder_list)
          self.decoder_list=nn.ModuleList(decoder_list)

          self.fc=nn.Sequential(
              nn.Linear(6144,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
              nn.Linear(256,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
              nn.Linear(256,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
              nn.Linear(256,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
              nn.Linear(256,256),
              nn.InstanceNorm1d(256),
              nn.ReLU(),
          )

          self.criterion=nn.L1Loss()
          
    def forward(self,bat_input_texture):
        inpaint_texture_list=[]
        global_code_list=[]
        project_code_list=[]
        for k in range(bat_input_texture.shape[1]):
            code_list=[]
            for i in range(4):
                for j in range(6):
                    body_part_texture=bat_input_texture[:,k,:,i*200:i*200+200,j*200:j*200+200]
                    encoder_index=i*6+j
                    encoder=self.encoder_list[encoder_index]
                    code_list.append(encoder(body_part_texture))
            global_code=torch.cat(code_list,dim=2)#global code is 6144
            global_code_list.append(global_code)
            #print("global_code's shape is",global_code.shape)
            project_code=self.fc(global_code) # project code is 256
            project_code_list.append(project_code.unsqueeze(1))
        
        
        bat_global_code=torch.cat(global_code_list,dim=1)
        bat_project_code=torch.cat(project_code_list,dim=1)
        
        fus_global_code=torch.max(bat_global_code,dim=1)[0]
        fus_project_code=torch.max(bat_project_code,dim=1)[0]
        
        #print(fus_project_code.shape)
        
        fus_part_code=fus_global_code.view(bat_global_code.shape[0],24,-1)
            
        for i in range(4):
            for j in range(6):
                decoder_index=i*6+j
                decoder=self.decoder_list[decoder_index]
                input_code=torch.cat([fus_project_code.squeeze(1),fus_part_code[:,decoder_index].squeeze(1)],dim=1)#input code 512
                inpaint_texture_list.append(decoder(input_code))
                
        return inpaint_texture_list
        
# ResnetBlock from vid2vid
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


class PredictiveModule(nn.Module):
    def __init__(self):
        super(PredictiveModule, self).__init__()
        self.n_blocks = 6

        # in size: 256 x 256 x 9; out size: 64 x 64 x 256
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1),  # 256
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 64
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        padding_type = "zero"
        norm = nn.InstanceNorm2d
        activation = nn.ReLU(inplace=True)
        blocks = []
        for i in range(self.n_blocks):
            blocks += [ResnetBlock(256, padding_type=padding_type, activation=activation, norm_layer=norm)]
        self.res_blocks = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 128
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 256
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        return self.decoder(x)
        
class BlendingModule(nn.Module):
    def __init__(self):
        super(BlendingModule, self).__init__()
        self.n_blocks = 3

       
        padding_type = "zero"
        norm = nn.InstanceNorm2d
        activation = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1),  # 256
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm),
            ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm),
            ResnetBlock(64, padding_type=padding_type, activation=activation, norm_layer=norm),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # 128
            nn.Tanh()
        )
        self.criterion = nn.L1Loss()
        self.perceptual_loss=VGGLoss_CRN()
        '''
        blocks = []
        for i in range(self.n_blocks):
            blocks += [ResnetBlock(128, padding_type=padding_type, activation=activation, norm_layer=norm)]
        self.res_blocks = nn.Sequential(*blocks)
        '''

    def forward(self, predictive_output, warp_output,tgt_IUV,real):
        input = torch.cat([predictive_output, warp_output,tgt_IUV], dim=1)

        x = self.conv(input)+predictive_output

        l1_loss = self.criterion(vgg_preprocess(x), vgg_preprocess(real))
        perceptual_loss= self.perceptual_loss(vgg_preprocess(x), vgg_preprocess(real))
       
        #print("x's shape",x.shape)
        #print("residual's shape",residual.shape)
        #print("predictive_output's shape is",predictive_output.shape)

        return x,l1_loss+perceptual_loss
        
class Downsampler(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=1, padding=1):
        super(Downsampler, self).__init__()
        self.enconv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.enconv(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, padding=1):
        super(Upsampler, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.myconv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, enc_x):
        x = self.up(x)
        x = torch.cat([x, enc_x], 1)
        x = self.myconv(x)
        return x
        
class Upsampler_SE(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, padding=1,output_size=50):
        super(Upsampler_SE, self).__init__()
        self.up = nn.UpsamplingBilinear2d(size=(output_size,output_size))
        self.myconv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, enc_x):
        x = self.up(x)
        x = torch.cat([x, enc_x], 1)
        x = self.myconv(x)
        return x
        
class UNet(nn.Module):
    def __init__(self, input_nc, enc_nc, dec_nc):
        super(UNet, self).__init__()

        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=7, padding=3)  # 256
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 128
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 64
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 32
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 16
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8])
        self.enc10 = Downsampler(enc_nc[8], enc_nc[9], stride=2)  # 8
        self.enc11 = Downsampler(enc_nc[9], enc_nc[10])

        self.dec1 = Upsampler(enc_nc[8] + enc_nc[10], dec_nc[0])
        self.dec2 = Upsampler(enc_nc[6] + dec_nc[0], dec_nc[1])
        self.dec3 = Upsampler(enc_nc[4] + dec_nc[1], dec_nc[2])
        self.dec4 = Upsampler(enc_nc[2] + dec_nc[2], dec_nc[3])
        self.dec5 = Upsampler(enc_nc[0] + dec_nc[3], dec_nc[4])
        
        self.conv = nn.Conv2d(dec_nc[4], 3, kernel_size=3, padding=1)

    def forward(self, x_in):
        x0 = self.enc1(x_in)
        x1 = self.enc2(x0)
        x3 = self.enc3(x1)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)
        x10 = self.enc10(x9)
        x = self.enc11(x10)

        x = self.dec1(x, x9)
        x = self.dec2(x, x7)
        x = self.dec3(x, x5)
        x = self.dec4(x, x3)
        x = self.dec5(x, x0)
        
        x=self.conv(x)

        return x
        
class UNet_TA(nn.Module):  #input two texture map, -1x6x800x1200,output one texture map -1x3x800x1200
    def __init__(self, input_nc, enc_nc, dec_nc):
        super(UNet_TA, self).__init__()

        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=7, padding=3)  # 800 1200
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 400 600
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 200 300
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 100 150
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 50 75
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8])

        self.dec1 = Upsampler(enc_nc[6] + enc_nc[4], dec_nc[0],output_size=25) # 
        self.dec2 = Upsampler(enc_nc[4] + dec_nc[0], dec_nc[1],output_size=50)
        self.dec3 = Upsampler(enc_nc[2] + dec_nc[1], dec_nc[2],output_size=100)
        self.dec4 = Upsampler(enc_nc[0] + dec_nc[2], dec_nc[3],output_size=200)
        self.conv = nn.Conv2d(dec_nc[3], 3, kernel_size=3, padding=1)
        self.criterion = nn.L1Loss()
        self.perceptual_loss=VGGLoss_CRN()

    def forward(self, x_in,src_texture_mask,tgt_texture_mask,tgt_texture_im):
        x0 = self.enc1(x_in)
        x1 = self.enc2(x0)
        x3 = self.enc3(x1)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)

        x = self.dec1(x9, x7)
        x = self.dec2(x, x5)
        x = self.dec3(x, x3)
        x = self.dec4(x, x0)
        
        x = self.conv(x) #output should be 3,800,1200
        
        l1_loss=0
        #per_loss=0
        common_src_area=torch.zeros(src_texture_mask[:,0].squeeze(1).shape,dtype=torch.uint8).cuda()
        for i in range(src_texture_mask.shape[1]):
            common_src_area=common_src_area | src_texture_mask[:,i].squeeze(1)
        for i in range(tgt_texture_mask.shape[1]):
            loss_area=common_src_area & tgt_texture_mask[:,i].squeeze(1)
            loss_area=loss_area.float()
            gen_loss_area=loss_area*x
            real_loss_area=loss_area*tgt_texture_im[:,0].float().squeeze(1)
            l1_loss=l1_loss+self.criterion(gen_loss_area,real_loss_area)
            #per_loss=per_loss+self.perceptual_loss(vgg_preprocess(gen_loss_area),vgg_preprocess(real_loss_area))
        

        return x,l1_loss
        
class UNet_SE(nn.Module):  #input two texture map, -1x72x200x200,output one texture map -1x3x800x1200
    def __init__(self, input_nc, enc_nc, dec_nc):
        super(UNet_SE, self).__init__()

        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=5, padding=2)  # 12 200 200
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 24 100 100
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 24 50 50
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 48 25 25
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 96 12 12
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8]) 

        self.dec1 = Upsampler_SE(enc_nc[8] + enc_nc[6], dec_nc[0],output_size=25) #48 25 25 
        self.dec2 = Upsampler_SE(enc_nc[4] + dec_nc[0], dec_nc[1],output_size=50)#24 50 50
        self.dec3 = Upsampler_SE(enc_nc[2] + dec_nc[1], dec_nc[2],output_size=100)# 12 100 100
        self.dec4 = Upsampler_SE(enc_nc[0] + dec_nc[2], dec_nc[3],output_size=200)# 6 200 200
        self.conv = nn.Conv2d(dec_nc[3], 3, kernel_size=3, padding=1)# 3 200 200
        #self.criterion = nn.L1Loss()
        #self.perceptual_loss=VGGLoss_CRN()

    def forward(self, x_in):
        x0 = self.enc1(x_in)
        x1 = self.enc2(x0)
        x3 = self.enc3(x1)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)

        x = self.dec1(x9, x7)
        x = self.dec2(x, x5)
        x = self.dec3(x, x3)
        x = self.dec4(x, x0)
        
        x = self.conv(x) #output should be 36,200,200
        #x = x.view(-1,3,800,120

        return x

class Accumulate(nn.Module):  #input two texture map, -1x144x200x200 (store in a list),output one texture map -1x3x800x1200
    def __init__(self,in_channel=6):
        super(Accumulate, self).__init__()
        self.UNet_SE_list=[]
        for i in range(24):
            self.UNet_SE_list.append(UNet_SE(in_channel, [12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.UNet_SE_list=nn.ModuleList(self.UNet_SE_list)
        
        self.criterion = nn.L1Loss()
        #self.perceptual_loss=VGGLoss_CRN()
        
        
    def forward(self, x_in,src_texture_mask,tgt_texture_mask,tgt_texture_im): # x_in is a list,each element is 6*200*200
        texture_list=[]
        for ind in range(24):
            #print(x_in[ind].shape)
            texture_list.append(self.UNet_SE_list[ind](x_in[ind])) 
        texture_image=torch.zeros(())
        texture_image=texture_image.new_empty((src_texture_mask.shape[0],3,800,1200)).cuda()
        for i in range(4):
            for j in range(6):
                #print(i*6+j)
                #print(texture_list[i*6+j].shape)
                texture_image[:,:,i*200:(i+1)*200,j*200:(j+1)*200]=texture_list[i*6+j]
                
        l1_loss=0
        #per_loss=0
        common_src_area=torch.zeros(src_texture_mask[:,0].squeeze(1).shape,dtype=torch.uint8).cuda()
        for i in range(src_texture_mask.shape[1]):
            common_src_area=common_src_area | src_texture_mask[:,i].squeeze(1)
        #common src area should be -1 3 800 1200
        #tgt texture mask is -1 3 3 800 1200
        for i in range(tgt_texture_mask.shape[1]):
            loss_area=common_src_area & tgt_texture_mask[:,i].squeeze(1)
            loss_area=loss_area.float()
            gen_loss_area=loss_area*texture_image
            real_loss_area=loss_area*tgt_texture_im[:,0].float().squeeze(1)
            l1_loss=l1_loss+self.criterion(gen_loss_area,real_loss_area)
            #per_loss=per_loss+self.perceptual_loss(vgg_preprocess(gen_loss_area),vgg_preprocess(real_loss_area))
        
        total_loss=l1_loss
        
        return texture_image,total_loss
        
class Accumulate_no_loss(nn.Module):  #input two texture map, -1x144x200x200 (store in a list),output one texture map -1x3x800x1200
    def __init__(self,in_channel=6):
        super(Accumulate_no_loss, self).__init__()
        self.UNet_SE_list=[]
        for i in range(24):
            self.UNet_SE_list.append(UNet_SE(in_channel, [12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.UNet_SE_list=nn.ModuleList(self.UNet_SE_list)
        
        self.criterion = nn.L1Loss()
        #self.perceptual_loss=VGGLoss_CRN()
        
        
    def forward(self, x_in): # x_in is a list,each element is 6*200*200
        texture_list=[]
        for ind in range(24):
            #print(x_in[ind].shape)
            texture_list.append(self.UNet_SE_list[ind](x_in[ind])) 
        
        return texture_list
   
class Downsampler_stack(nn.Module):
    def __init__(self, input_nc, enc_nc,compress_nc):
        super(Downsampler_stack, self).__init__()
        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=5, padding=2)  # 12 200 200
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 24 100 100
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 24 50 50
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 48 25 25
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 96 12 12
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8]) 
        
        self.enc_compress=Downsampler(enc_nc[8],compress_nc) # 3 12 12
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)
        
        x_compress=self.enc_compress(x9)
        
        encode_list=[x1,x3,x5,x7,x9]

        return encode_list,x_compress

class Upsampler_stack(nn.Module):
    def __init__(self, input_nc, enc_nc,embed_nc,dec_nc):
        super(Upsampler_stack, self).__init__()
        self.dec1 = Upsampler_SE(enc_nc[8] + enc_nc[6]+embed_nc, dec_nc[0],output_size=25) #48 25 25 
        self.dec2 = Upsampler_SE(enc_nc[4] + dec_nc[0], dec_nc[1],output_size=50)#24 50 50
        self.dec3 = Upsampler_SE(enc_nc[2] + dec_nc[1], dec_nc[2],output_size=100)# 12 100 100
        self.dec4 = Upsampler_SE(enc_nc[0] + dec_nc[2], dec_nc[3],output_size=200)# 6 200 200
        self.conv = nn.Conv2d(dec_nc[3], 3, kernel_size=3, padding=1)# 3 200 200

    def forward(self, global_embed,encode_vector):
        #print(global_embed.shape,encode_vector[4].shape)
        x_input=torch.cat([encode_vector[4],global_embed],dim=1)
        x = self.dec1(x_input,encode_vector[3])
        x = self.dec2(x,encode_vector[2])
        x = self.dec3(x,encode_vector[1])
        x = self.dec4(x,encode_vector[0])
        x = self.conv(x)
        return x
        
class Downsampler_stack_noEmbed(nn.Module):
    def __init__(self, input_nc, enc_nc):
        super(Downsampler_stack_noEmbed, self).__init__()
        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=5, padding=2)  # 12 200 200
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 24 100 100
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 24 50 50
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 48 25 25
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 96 12 12
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8]) 
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)
        
        return x1.unsqueeze(1),x3.unsqueeze(1),x5.unsqueeze(1),x7.unsqueeze(1),x9.unsqueeze(1)
        
class Upsampler_stack_noEmbed(nn.Module):
    def __init__(self, enc_nc,dec_nc):
        super(Upsampler_stack_noEmbed, self).__init__()
        self.dec1 = Upsampler_SE(enc_nc[8] + enc_nc[6], dec_nc[0],output_size=25) #48 25 25 
        self.dec2 = Upsampler_SE(enc_nc[4] + dec_nc[0], dec_nc[1],output_size=50)#24 50 50
        self.dec3 = Upsampler_SE(enc_nc[2] + dec_nc[1], dec_nc[2],output_size=100)# 12 100 100
        self.dec4 = Upsampler_SE(enc_nc[0] + dec_nc[2], dec_nc[3],output_size=200)# 6 200 200
        self.conv = nn.Conv2d(dec_nc[3], 3, kernel_size=3, padding=1)# 3 200 200

    def forward(self, encode_vector):
        x_input=encode_vector[4]
        x = self.dec1(x_input,encode_vector[3])
        x = self.dec2(x,encode_vector[2])
        x = self.dec3(x,encode_vector[1])
        x = self.dec4(x,encode_vector[0])
        x = self.conv(x)
        return x
        
class Downsampler_mask(nn.Module):
    def __init__(self, input_nc, enc_nc):
        super(Downsampler_mask, self).__init__()
        self.enc=enc_nc
        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=5, padding=2)  # 12 200 200
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 24 100 100
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 24 50 50
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 48 25 25
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 96 12 12
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8]) 
        
        self.mask1 = nn.Sequential(
            nn.Conv2d(in_channels=enc_nc[0]*3, out_channels=3, kernel_size=5, padding=2),
            nn.Softmax(dim=1))
        self.mask2 = nn.Sequential(
            nn.Conv2d(in_channels=enc_nc[2]*3, out_channels=3, kernel_size=3, padding=1),
            nn.Softmax(dim=1))
        self.mask3 = nn.Sequential(
            nn.Conv2d(in_channels=enc_nc[4]*3, out_channels=3, kernel_size=3, padding=1),
            nn.Softmax(dim=1))
        self.mask4 = nn.Sequential(
            nn.Conv2d(in_channels=enc_nc[6]*3, out_channels=3, kernel_size=3, padding=1),
            nn.Softmax(dim=1))
        self.mask5 = nn.Sequential(
            nn.Conv2d(in_channels=enc_nc[8]*3, out_channels=3, kernel_size=3, padding=1),
            nn.Softmax(dim=1))
        
    def forward(self, x): #input is a list with three elements
        x_input=torch.cat(x,dim=0)
        batch_size=x[0].shape[0]
        x1 = self.enc1(x_input)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)
        
        x1_con=torch.cat([x1[batch_size*0:batch_size*1],x1[batch_size*1:batch_size*2],x1[batch_size*2:batch_size*3]],dim=1)
        x3_con=torch.cat([x3[batch_size*0:batch_size*1],x3[batch_size*1:batch_size*2],x3[batch_size*2:batch_size*3]],dim=1)
        x5_con=torch.cat([x5[batch_size*0:batch_size*1],x5[batch_size*1:batch_size*2],x5[batch_size*2:batch_size*3]],dim=1)
        x7_con=torch.cat([x7[batch_size*0:batch_size*1],x7[batch_size*1:batch_size*2],x7[batch_size*2:batch_size*3]],dim=1)
        x9_con=torch.cat([x9[batch_size*0:batch_size*1],x9[batch_size*1:batch_size*2],x9[batch_size*2:batch_size*3]],dim=1)
        x1_mask=self.mask1(x1_con)
        x1_mask=torch.cat([x1_mask[:,0].unsqueeze(1).repeat(1,12,1,1),x1_mask[:,1].unsqueeze(1).repeat(1,12,1,1),x1_mask[:,2].unsqueeze(1).repeat(1,12,1,1)],dim=1)
        #print(x1_mask.shape)
        x3_mask=self.mask2(x3_con)
        x3_mask=torch.cat([x3_mask[:,0].unsqueeze(1).repeat(1,24,1,1),x3_mask[:,1].unsqueeze(1).repeat(1,24,1,1),x3_mask[:,2].unsqueeze(1).repeat(1,24,1,1)],dim=1)
        x5_mask=self.mask3(x5_con)
        x5_mask=torch.cat([x5_mask[:,0].unsqueeze(1).repeat(1,24,1,1),x5_mask[:,1].unsqueeze(1).repeat(1,24,1,1),x5_mask[:,2].unsqueeze(1).repeat(1,24,1,1)],dim=1)
        x7_mask=self.mask4(x7_con)
        x7_mask=torch.cat([x7_mask[:,0].unsqueeze(1).repeat(1,48,1,1),x7_mask[:,1].unsqueeze(1).repeat(1,48,1,1),x7_mask[:,2].unsqueeze(1).repeat(1,48,1,1)],dim=1)
        x9_mask=self.mask5(x9_con)
        x9_mask=torch.cat([x9_mask[:,0].unsqueeze(1).repeat(1,96,1,1),x9_mask[:,1].unsqueeze(1).repeat(1,96,1,1),x9_mask[:,2].unsqueeze(1).repeat(1,96,1,1)],dim=1)
        #print(x1_con.shape,x1_mask.shape)
        x1_con=x1_con*x1_mask
        x3_con=x3_con*x3_mask
        x5_con=x5_con*x5_mask
        x7_con=x7_con*x7_mask
        x9_con=x9_con*x9_mask
        
        x1_con=x1_con[:,0:12]+x1_con[:,12:24]+x1_con[:,24:36]
        x3_con=x3_con[:,0:24]+x3_con[:,24:48]+x3_con[:,48:72]
        x5_con=x5_con[:,0:24]+x5_con[:,24:48]+x5_con[:,48:72]
        x7_con=x7_con[:,0:48]+x7_con[:,48:96]+x7_con[:,96:144]
        x9_con=x9_con[:,0:96]+x9_con[:,96:192]+x9_con[:,192:288] 
        
        return x1_con,x3_con,x5_con,x7_con,x9_con
        
class Downsampler_convLSTM(nn.Module):
    def __init__(self, input_nc, enc_nc):
        super(Downsampler_convLSTM, self).__init__()
        self.enc=enc_nc
        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=5, padding=2)  # 12 200 200
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 24 100 100
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 24 50 50
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 48 25 25
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 96 12 12
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8]) 
        
        self.convLSTM1 = ConvLSTM((200,200), enc_nc[0], [enc_nc[0]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convLSTM2 = ConvLSTM((100,100), enc_nc[2], [enc_nc[2]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convLSTM3 = ConvLSTM((50,50), enc_nc[4], [enc_nc[4]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convLSTM4 = ConvLSTM((25,25), enc_nc[6], [enc_nc[6]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convLSTM5 = ConvLSTM((13,13), enc_nc[8], [enc_nc[8]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)

        
    def forward(self, x): #input is a list with three elements
        x_input=torch.cat(x,dim=0)
        batch_size=x[0].shape[0]
        x1 = self.enc1(x_input)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)
        seq_len=len(x)
        x1_list=[]
        x3_list=[]
        x5_list=[]
        x7_list=[]
        x9_list=[]
        for i in range(seq_len):
            x1_list.append(x1[batch_size*i:batch_size*(i+1)])
            x3_list.append(x3[batch_size*i:batch_size*(i+1)])
            x5_list.append(x5[batch_size*i:batch_size*(i+1)])
            x7_list.append(x7[batch_size*i:batch_size*(i+1)])
            x9_list.append(x9[batch_size*i:batch_size*(i+1)])
        x1_con=torch.stack(x1_list,dim=1)
        x3_con=torch.stack(x3_list,dim=1)
        x5_con=torch.stack(x5_list,dim=1)
        x7_con=torch.stack(x7_list,dim=1)
        x9_con=torch.stack(x9_list,dim=1)
        
        _,x1_last_state_list=self.convLSTM1(x1_con)
        x1=x1_last_state_list[-1][0]
        _,x3_last_state_list=self.convLSTM2(x3_con)
        x3=x3_last_state_list[-1][0]
        _,x5_last_state_list=self.convLSTM3(x5_con)
        x5=x5_last_state_list[-1][0]
        _,x7_last_state_list=self.convLSTM4(x7_con)
        x7=x7_last_state_list[-1][0]
        _,x9_last_state_list=self.convLSTM5(x9_con)
        x9=x9_last_state_list[-1][0]
        
        return x1,x3,x5,x7,x9
        
class Downsampler_GRU(nn.Module):
    def __init__(self, input_nc, enc_nc):
        super(Downsampler_GRU, self).__init__()
        self.enc=enc_nc
        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=5, padding=2)  # 12 200 200
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 24 100 100
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 24 50 50
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 48 25 25
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 96 12 12
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8]) 
        
        self.convGRU1 = convGRU((200,200), enc_nc[0], [enc_nc[0]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convGRU2 = convGRU((100,100), enc_nc[2], [enc_nc[2]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convGRU3 = convGRU((50,50), enc_nc[4], [enc_nc[4]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convGRU4 = convGRU((25,25), enc_nc[6], [enc_nc[6]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convGRU5 = convGRU((13,13), enc_nc[8], [enc_nc[8]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        
    def forward(self, x): #input is a list with three elements
        x_input=torch.cat(x,dim=0)
        batch_size=x[0].shape[0]
        #print(x_input.shape)
        x1 = self.enc1(x_input)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)
        
        seq_len=len(x)
        x1_list=[]
        x3_list=[]
        x5_list=[]
        x7_list=[]
        x9_list=[]
        for i in range(seq_len):
            x1_list.append(x1[batch_size*i:batch_size*(i+1)])
            x3_list.append(x3[batch_size*i:batch_size*(i+1)])
            x5_list.append(x5[batch_size*i:batch_size*(i+1)])
            x7_list.append(x7[batch_size*i:batch_size*(i+1)])
            x9_list.append(x9[batch_size*i:batch_size*(i+1)])
        x1_con=torch.stack(x1_list,dim=1)
        x3_con=torch.stack(x3_list,dim=1)
        x5_con=torch.stack(x5_list,dim=1)
        x7_con=torch.stack(x7_list,dim=1)
        x9_con=torch.stack(x9_list,dim=1)
        
        _,x1_last_state_list=self.convGRU1(x1_con)
        x1=x1_last_state_list[-1]
        _,x3_last_state_list=self.convGRU2(x3_con)
        x3=x3_last_state_list[-1]
        _,x5_last_state_list=self.convGRU3(x5_con)
        x5=x5_last_state_list[-1]
        _,x7_last_state_list=self.convGRU4(x7_con)
        x7=x7_last_state_list[-1]
        _,x9_last_state_list=self.convGRU5(x9_con)
        x9=x9_last_state_list[-1]
        
        return x1,x3,x5,x7,x9
        
class Downsampler_ModGRU(nn.Module):
    def __init__(self, input_nc, enc_nc):
        super(Downsampler_ModGRU, self).__init__()
        self.enc=enc_nc
        self.enc1 = Downsampler(input_nc, enc_nc[0], kernel_size=5, padding=2)  # 12 200 200
        self.enc2 = Downsampler(enc_nc[0], enc_nc[1], stride=2)  # 24 100 100
        self.enc3 = Downsampler(enc_nc[1], enc_nc[2])
        self.enc4 = Downsampler(enc_nc[2], enc_nc[3], stride=2)  # 24 50 50
        self.enc5 = Downsampler(enc_nc[3], enc_nc[4])
        self.enc6 = Downsampler(enc_nc[4], enc_nc[5], stride=2)  # 48 25 25
        self.enc7 = Downsampler(enc_nc[5], enc_nc[6])
        self.enc8 = Downsampler(enc_nc[6], enc_nc[7], stride=2)  # 96 12 12
        self.enc9 = Downsampler(enc_nc[7], enc_nc[8]) 
        
        self.convGRU1 = ModconvGRU((200,200), enc_nc[0], [enc_nc[0]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convGRU2 = ModconvGRU((100,100), enc_nc[2], [enc_nc[2]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convGRU3 = ModconvGRU((50,50), enc_nc[4], [enc_nc[4]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convGRU4 = ModconvGRU((25,25), enc_nc[6], [enc_nc[6]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        self.convGRU5 = ModconvGRU((13,13), enc_nc[8], [enc_nc[8]], [(3,3)], 1,
                 batch_first=True, bias=True, return_all_layers=False)
        
    def forward(self, x): #input is a list with three elements
        x_input=torch.cat(x,dim=0)
        batch_size=x[0].shape[0]
        #print(x_input.shape)
        x1 = self.enc1(x_input)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        x9 = self.enc9(x8)
        
        seq_len=len(x)
        x1_list=[]
        x3_list=[]
        x5_list=[]
        x7_list=[]
        x9_list=[]
        for i in range(seq_len):
            x1_list.append(x1[batch_size*i:batch_size*(i+1)])
            x3_list.append(x3[batch_size*i:batch_size*(i+1)])
            x5_list.append(x5[batch_size*i:batch_size*(i+1)])
            x7_list.append(x7[batch_size*i:batch_size*(i+1)])
            x9_list.append(x9[batch_size*i:batch_size*(i+1)])
        x1_con=torch.stack(x1_list,dim=1)
        x3_con=torch.stack(x3_list,dim=1)
        x5_con=torch.stack(x5_list,dim=1)
        x7_con=torch.stack(x7_list,dim=1)
        x9_con=torch.stack(x9_list,dim=1)
        
        _,x1_last_state_list=self.convGRU1(x1_con)
        x1=x1_last_state_list[-1]
        _,x3_last_state_list=self.convGRU2(x3_con)
        x3=x3_last_state_list[-1]
        _,x5_last_state_list=self.convGRU3(x5_con)
        x5=x5_last_state_list[-1]
        _,x7_last_state_list=self.convGRU4(x7_con)
        x7=x7_last_state_list[-1]
        _,x9_last_state_list=self.convGRU5(x9_con)
        x9=x9_last_state_list[-1]
        
        return x1,x3,x5,x7,x9
        
class Accumulate_GRU(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_LSTM,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        self.mask_conv_list=[]
        enc=[12,24,24,48,96]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_GRU(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        self.criterion = nn.L1Loss()
        
    def forward(self, x_in,src_texture_mask,tgt_texture_mask,tgt_texture_im): # x_in is a two_dim list,x_in[ind] is the same part from three textures
        texture_list=[]
        for ind in range(24):
            x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[ind])
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
            
        texture_image=torch.zeros(())
        texture_image=texture_image.new_empty((src_texture_mask.shape[0],3,800,1200)).cuda()
        for i in range(4):
            for j in range(6):
                #print(i*6+j)
                #print(texture_list[i*6+j].shape)
                texture_image[:,:,i*200:(i+1)*200,j*200:(j+1)*200]=texture_list[i*6+j]
                
        l1_loss=0
        #per_loss=0
        common_src_area=torch.zeros(src_texture_mask[:,0].squeeze(1).shape,dtype=torch.uint8).cuda()
        for i in range(src_texture_mask.shape[1]):
            common_src_area=common_src_area | src_texture_mask[:,i].squeeze(1)
        #common src area should be -1 3 800 1200
        #tgt texture mask is -1 3 3 800 1200
        for i in range(tgt_texture_mask.shape[1]):
            loss_area=common_src_area & tgt_texture_mask[:,i].squeeze(1)
            loss_area=loss_area.float()
            gen_loss_area=loss_area*texture_image
            real_loss_area=loss_area*tgt_texture_im[:,0].float().squeeze(1)
            l1_loss=l1_loss+self.criterion(gen_loss_area,real_loss_area)
            #per_loss=per_loss+self.perceptual_loss(vgg_preprocess(gen_loss_area),vgg_preprocess(real_loss_area))
        
        total_loss=l1_loss
    
        return texture_image,total_loss
        
class Accumulate_ModGRU_no_loss(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_ModGRU_no_loss,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        self.mask_conv_list=[]
        enc=[12,24,24,48,96]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_ModGRU(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        self.criterion = nn.L1Loss()
        
    def forward(self, x_in): # x_in is a two_dim list,x_in[ind] is the same part from three textures
        texture_list=[]
        for ind in range(24):
            x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[ind])
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
            
        return texture_list
        
class Accumulate_GRU_no_loss(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_GRU_no_loss,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        self.mask_conv_list=[]
        enc=[12,24,24,48,96]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_GRU(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        self.criterion = nn.L1Loss()
        
    def forward(self, x_in): # x_in is a two_dim list,x_in[ind] is the same part from three textures
        texture_list=[]
        for ind in range(24):
            x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[ind])
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
            
        return texture_list
        
class Accumulate_LSTM(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_LSTM,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        self.mask_conv_list=[]
        enc=[12,24,24,48,96]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_convLSTM(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        self.criterion = nn.L1Loss()
        
    def forward(self, x_in,src_texture_mask,tgt_texture_mask,tgt_texture_im): # x_in is a two_dim list,x_in[ind] is the same part from three textures
        texture_list=[]
        for ind in range(24):
            x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[ind])
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
            
        texture_image=torch.zeros(())
        texture_image=texture_image.new_empty((src_texture_mask.shape[0],3,800,1200)).cuda()
        for i in range(4):
            for j in range(6):
                #print(i*6+j)
                #print(texture_list[i*6+j].shape)
                texture_image[:,:,i*200:(i+1)*200,j*200:(j+1)*200]=texture_list[i*6+j]
                
        l1_loss=0
        #per_loss=0
        common_src_area=torch.zeros(src_texture_mask[:,0].squeeze(1).shape,dtype=torch.uint8).cuda()
        for i in range(src_texture_mask.shape[1]):
            common_src_area=common_src_area | src_texture_mask[:,i].squeeze(1)
        #common src area should be -1 3 800 1200
        #tgt texture mask is -1 3 3 800 1200
        for i in range(tgt_texture_mask.shape[1]):
            loss_area=common_src_area & tgt_texture_mask[:,i].squeeze(1)
            loss_area=loss_area.float()
            gen_loss_area=loss_area*texture_image
            real_loss_area=loss_area*tgt_texture_im[:,0].float().squeeze(1)
            l1_loss=l1_loss+self.criterion(gen_loss_area,real_loss_area)
            #per_loss=per_loss+self.perceptual_loss(vgg_preprocess(gen_loss_area),vgg_preprocess(real_loss_area))
        
        total_loss=l1_loss
    
        return texture_image,total_loss
        
class Accumulate_LSTM_no_loss(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_LSTM_no_loss,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        self.mask_conv_list=[]
        enc=[12,24,24,48,96]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_convLSTM(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        self.criterion = nn.L1Loss()
        
    def forward(self, x_in): # x_in is a two_dim list,x_in[ind] is the same part from three textures
        texture_list=[]
        for ind in range(24):
            x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[ind])
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
    
        return texture_list
        
class Accumulate_mask(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_mask,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        self.mask_conv_list=[]
        enc=[12,24,24,48,96]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_mask(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        self.criterion = nn.L1Loss()
        
    def forward(self, x_in,src_texture_mask,tgt_texture_mask,tgt_texture_im): # x_in is a two_dim list,x_in[ind] is the same part from three textures
        texture_list=[]
        for ind in range(24):
            x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[ind])
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
            
        texture_image=torch.zeros(())
        texture_image=texture_image.new_empty((src_texture_mask.shape[0],3,800,1200)).cuda()
        for i in range(4):
            for j in range(6):
                #print(i*6+j)
                #print(texture_list[i*6+j].shape)
                texture_image[:,:,i*200:(i+1)*200,j*200:(j+1)*200]=texture_list[i*6+j]
                
        l1_loss=0
        #per_loss=0
        common_src_area=torch.zeros(src_texture_mask[:,0].squeeze(1).shape,dtype=torch.uint8).cuda()
        for i in range(src_texture_mask.shape[1]):
            common_src_area=common_src_area | src_texture_mask[:,i].squeeze(1)
        #common src area should be -1 3 800 1200
        #tgt texture mask is -1 3 3 800 1200
        for i in range(tgt_texture_mask.shape[1]):
            loss_area=common_src_area & tgt_texture_mask[:,i].squeeze(1)
            loss_area=loss_area.float()
            gen_loss_area=loss_area*texture_image
            real_loss_area=loss_area*tgt_texture_im[:,0].float().squeeze(1)
            l1_loss=l1_loss+self.criterion(gen_loss_area,real_loss_area)
            #per_loss=per_loss+self.perceptual_loss(vgg_preprocess(gen_loss_area),vgg_preprocess(real_loss_area))
        
        total_loss=l1_loss
    
        return texture_image,total_loss
        
class Accumulate_mask_no_loss(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_mask_no_loss,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        self.mask_conv_list=[]
        enc=[12,24,24,48,96]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_mask(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        self.criterion = nn.L1Loss()
        
    def forward(self, x_in): # x_in is a two_dim list,x_in[ind] is the same part from three textures
        texture_list=[]
        for ind in range(24): 
            x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[ind])
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
    
        return texture_list
        
class Accumulate_max_fusion(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_max_fusion,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_stack_noEmbed(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        
    def forward(self, x_in): # x_in is a multi-dimensional list,each element is 6*200*200
        texture_list=[]
        num_input=len(x_in)
        for ind in range(24):
            for i in range(num_input):
                if i==0:
                    x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[i][ind])
                else:
                    x1_n,x3_n,x5_n,x7_n,x9_n=self.Downsampler_list[ind](x_in[i][ind])
                    x1=torch.cat([x1,x1_n],dim=1)
                    x3=torch.cat([x3,x3_n],dim=1)
                    x5=torch.cat([x5,x5_n],dim=1)
                    x7=torch.cat([x7,x7_n],dim=1)
                    x9=torch.cat([x9,x9_n],dim=1)
            x1=torch.max(x1,1)[0]
            x3=torch.max(x3,1)[0]
            x5=torch.max(x5,1)[0]
            x7=torch.max(x7,1)[0]
            x9=torch.max(x9,1)[0]
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
    
        return texture_list
        
class Accumulate_avg_fusion(nn.Module):#UNet_SE(6, [12,24,24,24,24,48,48,96,96], [48,24,12,6])
    def __init__(self):
        super(Accumulate_avg_fusion,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_stack_noEmbed(3, [12,24,24,24,24,48,48,96,96]))
            self.Upsampler_list.append(Upsampler_stack_noEmbed([12,24,24,24,24,48,48,96,96], [48,24,12,6]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        
    def forward(self, x_in): # x_in is a multi-dimensional list,each element is 6*200*200
        texture_list=[]
        num_input=len(x_in)
        for ind in range(24):
            for i in range(num_input):
                if i==0:
                    x1,x3,x5,x7,x9=self.Downsampler_list[ind](x_in[i][ind])
                else:
                    x1_n,x3_n,x5_n,x7_n,x9_n=self.Downsampler_list[ind](x_in[i][ind])
                    x1=torch.cat([x1,x1_n],dim=1)
                    x3=torch.cat([x3,x3_n],dim=1)
                    x5=torch.cat([x5,x5_n],dim=1)
                    x7=torch.cat([x7,x7_n],dim=1)
                    x9=torch.cat([x9,x9_n],dim=1)
            x1=torch.mean(x1,1)
            x3=torch.mean(x3,1)
            x5=torch.mean(x5,1)
            x7=torch.mean(x7,1)
            x9=torch.mean(x9,1)
            encode_vector=[x1,x3,x5,x7,x9]
            texture_list.append(self.Upsampler_list[ind](encode_vector)) 
    
        return texture_list
        
class UNet_inpainter(nn.Module):
    def __init__(self):
        super(UNet_inpainter,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_stack(3, [12,24,24,24,24,48,48,96,96], 3))
            self.Upsampler_list.append(Upsampler_stack(3, [12,24,24,24,24,48,48,96,96],3*24, [96,48,24,12]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        
    def forward(self,texture_list):
        global_embed=[]
        encode_vector_list=[]
        output_texture_list=[]
        for i in range(24):
            encode_list,embed=self.Downsampler_list[i](texture_list[i])
            global_embed.append(embed)
            encode_vector_list.append(encode_list)
        global_embed=torch.cat(global_embed,dim=1) # it should be -1, 3*24,12,12
        for i in range(24):
            output_texture_list.append(self.Upsampler_list[i](global_embed,encode_vector_list[i]))
            
        return output_texture_list
        
class UNet_inpainter_varlen(nn.Module):
    def __init__(self,in_channel=9):
        super(UNet_inpainter_varlen,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_stack(in_channel, [12,24,24,24,24,48,48,96,96], 3))
            self.Upsampler_list.append(Upsampler_stack(3, [12,24,24,24,24,48,48,96,96],3*24, [96,48,24,12]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        
        self.criterion = nn.L1Loss()
        
    def forward(self,texture_list,tgt_texture_mask,tgt_texture_im):
        global_embed=[]
        encode_vector_list=[]
        output_texture_list=[]
        for i in range(24):
            encode_list,embed=self.Downsampler_list[i](texture_list[i])
            global_embed.append(embed)
            encode_vector_list.append(encode_list)
        global_embed=torch.cat(global_embed,dim=1) # it should be -1, 3*24,12,12
        for i in range(24):
            output_texture_list.append(self.Upsampler_list[i](global_embed,encode_vector_list[i]))
        
        texture_image=torch.zeros(())
        texture_image=texture_image.new_empty((tgt_texture_mask.shape[0],3,800,1200)).cuda()
        for i in range(4):
            for j in range(6):
                #print(i*6+j)
                #print(texture_list[i*6+j].shape)
                texture_image[:,:,i*200:(i+1)*200,j*200:(j+1)*200]=output_texture_list[i*6+j]
                
        l1_loss=0
        
        for i in range(tgt_texture_mask.shape[1]):
            #print("using the correct mask")
            gen_loss_area=tgt_texture_mask[:,i].squeeze(1)*texture_image
            real_loss_area=tgt_texture_mask[:,i].squeeze(1)*tgt_texture_im[:,0].float().squeeze(1)
            l1_loss=l1_loss+self.criterion(gen_loss_area,real_loss_area)
            #per_loss=per_loss+self.perceptual_loss(vgg_preprocess(gen_loss_area),vgg_preprocess(real_loss_area))
        
        total_loss=l1_loss
        
        return texture_image,total_loss
        
class UNet_inpainter_varlen_no_loss(nn.Module):
    def __init__(self,in_channel=9):
        super(UNet_inpainter_varlen_no_loss,self).__init__()
        self.Downsampler_list=[]
        self.Upsampler_list=[]
        for i in range(24):
            self.Downsampler_list.append(Downsampler_stack(in_channel, [12,24,24,24,24,48,48,96,96], 3))
            self.Upsampler_list.append(Upsampler_stack(3, [12,24,24,24,24,48,48,96,96],3*24, [96,48,24,12]))
        self.Downsampler_list=nn.ModuleList(self.Downsampler_list)
        self.Upsampler_list=nn.ModuleList(self.Upsampler_list)
        
        self.criterion = nn.L1Loss()
        
    def forward(self,texture_list):
        global_embed=[]
        encode_vector_list=[]
        output_texture_list=[]
        for i in range(24):
            encode_list,embed=self.Downsampler_list[i](texture_list[i])
            global_embed.append(embed)
            encode_vector_list.append(encode_list)
        global_embed=torch.cat(global_embed,dim=1) # it should be -1, 3*24,12,12
        for i in range(24):
            output_texture_list.append(self.Upsampler_list[i](global_embed,encode_vector_list[i]))
        
        return output_texture_list
        
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.Accu_model = Accumulate_LSTM_no_loss()
        model_save_dir="/home/haolin/texture_accumulation/checkpoints"
        Accu_model_dir=os.path.join(model_save_dir,'text_accu_LSTM_1010') 
        Accu_model_weight_dir=os.path.join(Accu_model_dir,"iter_40000.pth")
        self.Accu_model.load_state_dict(torch.load(Accu_model_weight_dir))
    
        self.inpaint_model = UNet_inpainter()
        inpaint_model_dir=os.path.join(model_save_dir,'text_accu_inpaint_0923')
        inpaint_model_weight_dir=os.path.join(inpaint_model_dir,"inpaint_iter_42000.pth")
        self.inpaint_model.load_state_dict(torch.load(inpaint_model_weight_dir))
        
        self.bg_model=CRN_smaller(3)
        bg_model_dir=os.path.join(model_save_dir,'inpaint_global_modGRU_1015')
        bg_model_weight_dir=os.path.join(bg_model_dir,"bg_iter_39000.pth")
        self.bg_model.load_state_dict(torch.load(bg_model_weight_dir))
        
        self.refine_model=CRN_smaller(3,fg=True)
        #refine_model_dir=os.path.join(opt['model_save_dir'],'inpaint_global_LSTM_5fr_1025')
        #refine_model_weight_dir=os.path.join(refine_model_dir,"refine_iter_3000.pth")
        #self.refine_model.load_state_dict(torch.load(refine_model_weight_dir))
        
    def forward(self,src_texture_im_input,src_mask_im,src_img,src_common_area,tgt_IUV255,bg_incomplete):
        Accu_output_texture=self.Accu_model(src_texture_im_input)
        tgt_IUV255=tgt_IUV255
                        
        src_common_area=(src_common_area*0).byte()
        for i in range(4):
            src_common_area=src_common_area|src_mask_im[:,i].byte()
                
        src_common_area=src_common_area.float()
        src_common_area=src_common_area.unsqueeze(1).repeat(1,3,1,1)
        
        for i in range(4):
            for j in range(6):
                common_area=src_common_area[:,:,i*200:(i+1)*200,j*200:(j+1)*200]
                Accu_output_texture[i*6+j]=Accu_output_texture[i*6+j]*common_area
        inpaint_texture=self.inpaint_model(Accu_output_texture)
        inpaint_warp=torch.full((src_img[:,0].squeeze(1).shape),0).cuda()
        for i in range(src_img.shape[0]):
            inpaint_texture_list=list(map(lambda inp: inp[i], inpaint_texture))
            inpaint_warp[i]=texture_warp_pytorch(inpaint_texture_list,tgt_IUV255[i])
    
        refine_output,fg_mask=self.refine_model(inpaint_warp,256)
        with torch.no_grad():
            bg_output=self.bg_model(bg_incomplete,256)
        final_output=refine_output*fg_mask.repeat(1,3,1,1)+bg_output*(1-fg_mask.repeat(1,3,1,1))
        
        return final_output,refine_output

def define_D(input_nc, ndf, netD, device,n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, device,init_type, init_gain)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, device,init_type='normal',init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).to(device)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
        
if __name__=="__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_array=torch.zeros((2,6,800,1200)).cuda()
    src_texture_mask=torch.zeros((2,2,3,800,1200),dtype=torch.uint8).cuda()
    tgt_texture_mask=torch.zeros((2,3,3,800,1200),dtype=torch.uint8).cuda()
    print(input_array.shape)
    model=UNet_TA(6, [64] * 2 + [128] * 9, [128] * 4 + [64]).to(device)
    output,l1_loss,per_loss=model(input_array,src_texture_mask,tgt_texture_mask)
    print(output.shape,l1_loss,per_loss)
    