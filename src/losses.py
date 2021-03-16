import torch
import torch.nn as nn
import torch.nn as nn
from torchvision.models import vgg19
from torch.nn import AvgPool2d

from src.utils import vgg_preprocess


class VidLoss(nn.Module):
    def __init__(self, loss_func, w_type):
        super(VidLoss, self).__init__()

        self.loss_func = loss_func
        self.w_type = w_type
        self.loss_weights = {
            "linear": lambda n_frames: [i * 2 / (n_frames + n_frames ** 2) for i in range(1, n_frames + 1)]
        }

    def forward(self, x_seq, y_seq):
        assert x_seq.size()[2] == y_seq.size()[2]
        loss = 0.
        num_frames = x_seq.size()[2]
        loss_weight = self.loss_weights[self.w_type](num_frames)
        for i in range(num_frames):
            w = loss_weight[i]
            loss += w * self.loss_func(x_seq[:, :, i, ...], y_seq[:, :, i, ...])

        return loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask, target * mask)
        return loss


class TruncVgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(TruncVgg19, self).__init__()
        self.vgg_model = vgg19(pretrained=True).features

        # replace max pooling with average pooling to eliminate grid effect
        mp_list = [4, 9, 18, 27, 36]
        for mp_idx in mp_list:
            self.vgg_model._modules[str(mp_idx)] = AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.extracted_layers = (lambda x: [str(i) for i in x])([1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26])

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


class PVGGLoss(nn.Module):
    def __init__(self, resp_weights, n_layers, reg=0.1):
        super(PVGGLoss, self).__init__()

        self.trunc_vgg = TruncVgg19()
        self.feat_weights = resp_weights
        self.n_layers = n_layers
        self.reg = reg

    def forward(self, pred_y, true_y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pred_y_feats = self.trunc_vgg(vgg_preprocess(pred_y))
        true_y_feats = self.trunc_vgg(vgg_preprocess(true_y))

        loss = None
        for j in range(self.n_layers):

            std = torch.from_numpy(self.feat_weights[str(j)][1]) + self.reg
            std = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(std, 0), 2), 3).to(device)
            d = true_y_feats[j].detach() - pred_y_feats[j]
            loss_j = torch.mean(torch.abs(torch.div(d, std)))

            if j == 0:
                loss = loss_j
            else:
                loss = torch.add(loss, loss_j)
        return loss / (self.n_layers * 1.0)


class PVGGLossNoNorm(nn.Module):
    def __init__(self, n_layers):
        super(PVGGLossNoNorm, self).__init__()

        self.trunc_vgg = TruncVgg19()
        self.n_layers = n_layers

    def forward(self, pred_y, true_y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        L1loss = nn.L1Loss().to(device)

        pred_y_feats = self.trunc_vgg(pred_y)
        true_y_feats = self.trunc_vgg(true_y)

        loss = 0
        for j in range(self.n_layers):
            loss += L1loss(pred_y_feats[j], true_y_feats[j]).item()
        return loss / (self.n_layers * 1.0)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out




