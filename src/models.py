import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.networks import InpaintGenerator, EdgeGenerator, Discriminator
# from dataset import Dataset
from src.loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, opt):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = opt
        self.iteration = 0

        self.gen_weights_path = os.path.join(opt['network_dir'], name + '_gen.pth')
        self.dis_weights_path = os.path.join(opt['network_dir'], name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.module.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.module.state_dict()
        }, self.dis_weights_path)


class InpaintingModel(BaseModel):
    def __init__(self, opt):
        super(InpaintingModel, self).__init__('InpaintingModel', opt)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=True)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type='nsgan')

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(0.0001),
            betas=(0.0, 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(0.0001) * float(0.1),
            betas=(0.0, 0.9)
        )

    def process(self, fake, iuv, fg_mask, real):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(fake, iuv, fg_mask)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = real
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * 0.1  # INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs * fg_mask, real * fg_mask) / torch.mean(fg_mask)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs * fg_mask, real * fg_mask)
        gen_content_loss = gen_content_loss * 0.1  # CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * fg_mask, real * fg_mask)
        gen_style_loss = gen_style_loss * 250  # STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, fake, iuv, fg_mask):
        # images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((fake, iuv), dim=1)
        outputs = self.generator(inputs)  # in: [rgb(3) + iuv(3)]
        return outputs * fg_mask

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModelWithbg(BaseModel):
    def __init__(self, opt):
        super(InpaintingModelWithbg, self).__init__('InpaintingModel', opt)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=True)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type='nsgan')

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(0.0001),
            betas=(0.0, 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(0.0001) * float(0.1),
            betas=(0.0, 0.9)
        )

    def process(self, fake, iuv, real):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(fake, iuv)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = real
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * 0.1  # INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, real)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, real)
        gen_content_loss = gen_content_loss * 0.1  # CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs, real)
        gen_style_loss = gen_style_loss * 250  # STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, fake, iuv):
        # images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((fake, iuv), dim=1)
        outputs = self.generator(inputs)  # in: [rgb(3) + iuv(3)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModelWithbgNoVGGLoss(BaseModel):
    def __init__(self, opt):
        super(InpaintingModelWithbgNoVGGLoss, self).__init__('InpaintingModel', opt)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=True)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

        l1_loss = nn.L1Loss()
        # perceptual_loss = PerceptualLoss()
        # style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type='nsgan')

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        # self.add_module('perceptual_loss', perceptual_loss)
        # self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(0.0001),
            betas=(0.0, 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(0.0001) * float(0.1),
            betas=(0.0, 0.9)
        )

    def process(self, fake, iuv, real):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(fake, iuv)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = real
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * 0.1  # INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, real)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        # gen_content_loss = self.perceptual_loss(outputs , real )
        # gen_content_loss = gen_content_loss * 0.1  # CONTENT_LOSS_WEIGHT
        # gen_loss += gen_content_loss

        # generator style loss
        # gen_style_loss = self.style_loss(outputs , real )
        # gen_style_loss = gen_style_loss * 250  # STYLE_LOSS_WEIGHT
        # gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            # ("l_per", gen_content_loss.item()),
            # ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, fake, iuv):
        # images_masked = (images * (1 - masks).float()) + masks
        inputs = torch.cat((fake, iuv), dim=1)
        outputs = self.generator(inputs)  # in: [rgb(3) + iuv(3)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()


class InpaintingModelWithbgNoVGGLossMasked(BaseModel):
    def __init__(self, opt):
        super(InpaintingModelWithbgNoVGGLossMasked, self).__init__('InpaintingModel', opt)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = InpaintGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=True)
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

        l1_loss = nn.L1Loss()
        # perceptual_loss = PerceptualLoss()
        # style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type='nsgan')

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        # self.add_module('perceptual_loss', perceptual_loss)
        # self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(0.0001),
            betas=(0.0, 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(0.0001) * float(0.1),
            betas=(0.0, 0.9)
        )

    def process(self, fake, iuv, real, inpaint_mask):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(fake, iuv, inpaint_mask)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = real
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * 0.1  # INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, real)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        # gen_content_loss = self.perceptual_loss(outputs , real )
        # gen_content_loss = gen_content_loss * 0.1  # CONTENT_LOSS_WEIGHT
        # gen_loss += gen_content_loss

        # generator style loss
        # gen_style_loss = self.style_loss(outputs , real )
        # gen_style_loss = gen_style_loss * 250  # STYLE_LOSS_WEIGHT
        # gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            # ("l_per", gen_content_loss.item()),
            # ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, fake, iuv, inpaint_mask):
        images_masked = (fake * (1 - inpaint_mask).float()) + inpaint_mask
        inputs = torch.cat((images_masked, iuv), dim=1)
        outputs = self.generator(inputs)  # in: [rgb(3) + iuv(3)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()
