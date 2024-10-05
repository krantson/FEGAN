import torch
# import torchgeometry
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import glob
import os
import shutil
import numpy as np
from torch.autograd import Variable
from math import exp
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
# from guided_filter_pytorch.guided_filter import GuidedFilter

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# Definition of Frequency Reconstruction Loss


class BgCut_Loss(nn.Module):
    def __init__(self, thre=0.5):
        super(BgCut_Loss, self).__init__()
        self.thre = thre

    def forward(self, input):
        x = torch.sort(torch.flatten(torch.sum(torch.abs(input), dim=1), 1), descending=False)[0][:, :int(64*192*self.thre)]
        return torch.mean(torch.std(x, dim=(-2, -1)))


class binary_cluster_loss(nn.Module):
    def __init__(self):
        super(binary_cluster_loss, self).__init__()

    def forward(self, input):
        input_flatten = input.view(input.shape[0], 3, -1)
        maxv = torch.max(input_flatten, dim=-1).unsqueeze(-1).unsqueeze(-1)
        minv = torch.min(input_flatten, dim=-1).unsqueeze(-1).unsqueeze(-1)
        (torch.abs(input - maxv), torch.abs(input - minv))



class Freq_Recon_loss(nn.Module):
    def __init__(self):
        super(Freq_Recon_loss, self).__init__()

    def forward(self, input, target):
        input_fft = fft_spectrum(input)
        target_fft = fft_spectrum(target)

        loss = torch.abs(torch.subtract(target_fft, input_fft))
        loss = torch.log(loss + 1)

        min, max = float(loss.min()), float(loss.max())
        loss.clamp_(min = min, max=max)
        loss.add_(-min).div_(max-min+1e-5)

        return torch.mean(loss)


# Definition of Perceptual Loss
class VGG_loss(torch.nn.Module):
    def __init__(self, vgg):
        super(VGG_loss, self).__init__()
        self.vgg = vgg
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
    
    def forward(self, input, target):
        img_vgg = vgg_preprocess(input)
        target_vgg = vgg_preprocess(target)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


# Definition of SSIM Loss
class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# Definition of Total Variation Loss
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, : h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, : w_x - 1]).sum()
        
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class MaskTVLoss(nn.Module):
    def __init__(self):
        super(MaskTVLoss, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x, clean):
        batch_size = x.size()[0]
        # h_x = x.size()[2]
        # w_x = x.size()[3] 

        # count_h = (x.size()[2] - 1) * x.size()[3]
        # count_w = x.size()[2] * (x.size()[3] - 1)

        mask1 = torch.zeros_like(x)
        mask2 = torch.zeros_like(x)
        mask1[torch.where(clean >  0.5)] = 1
        mask2[torch.where(clean <= 0.5)] = 1
        # NaN 之间不能用==比较，NaN不可以做任何逻辑运算
        mask1 = -self.maxpool(-mask1) # change to MinPool
        mask2 = -self.maxpool(-mask2)


        top = F.pad(x, (0, 0, 1, 0), mode='replicate')[:, :, :-1, :]
        bottom = F.pad(x, (0, 0, 0, 1), mode='replicate')[:, :, 1:, :]
        left = F.pad(x, (1, 0, 0, 0), mode='replicate')[:, :, :, :-1]
        right = F.pad(x, (0, 1, 0, 0), mode='replicate')[:, :, :, 1:]

        # tv =  torch.pow(x - top, 2) + torch.pow(x - bottom, 2) + torch.pow(x - left, 2) + torch.pow(x - right, 2)
        tv =  torch.abs(x - top) + torch.abs(x - bottom) + torch.abs(x - left) + torch.abs(x - right)

        mask_tv = (tv * mask1 + tv * mask2).mean()

        return mask_tv / batch_size


class MinimalTVLoss(nn.Module):
    def __init__(self):
        super(MinimalTVLoss, self).__init__()
    
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)

        top = F.pad(x, (0, 0, 1, 0), mode='replicate')[:, :, :-1, :]
        bottom = F.pad(x, (0, 0, 0, 1), mode='replicate')[:, :, 1:, :]
        left = F.pad(x, (1, 0, 0, 0), mode='replicate')[:, :, :, :-1]
        right = F.pad(x, (0, 1, 0, 0), mode='replicate')[:, :, :, 1:]

        top_tv =  torch.pow(x - top, 2)
        bottom_tv = torch.pow(x - bottom, 2)
        left_tv = torch.pow(x - left, 2)
        right_tv = torch.pow(x - right, 2)

        min_tv = torch.min(torch.min(top_tv, bottom_tv), torch.min(left_tv, right_tv)).mean()
        return min_tv


# Guided filter
# class GF(nn.Module):
#     # https://pypi.org/project/guided-filter-pytorch/
#     def __init__(self, r:int = 5, eps:float=2e-1):
#         super(GF, self).__init__()
#         self.g = GuidedFilter(r, eps)
    
#     def forward(self, x, y):
#         return self.g(x, y)


# Random Color Shift Algorithm before feeding input to texture discriminator
class ColorShift(nn.Module):
    def __init__(self, device):
        super(ColorShift, self).__init__()
        self.device = device
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        r1, g1, b1 = input[:,0,:,:].unsqueeze(1), input[:,1,:,:].unsqueeze(1), input[:,2,:,:].unsqueeze(1)
        r2, g2, b2 = target[:,0,:,:].unsqueeze(1), target[:,1,:,:].unsqueeze(1), target[:,2,:,:].unsqueeze(1)
        
        # uniform random values
        b_weight = torch.FloatTensor(1).uniform_(0.014, 0.214).to(self.device)
        r_weight = torch.FloatTensor(1).uniform_(0.199, 0.399).to(self.device)
        g_weight = torch.FloatTensor(1).uniform_(0.487, 0.687).to(self.device)

        output1 = (b_weight*b1+g_weight*g1+r_weight*r1)/(b_weight+g_weight+r_weight)
        output2 = (b_weight*b2+g_weight*g2+r_weight*r2)/(b_weight+g_weight+r_weight)
        return output1, output2


class AdaptiveColorShift(nn.Module):
    """
    随机颜色增强
    """
    def __init__(self, device):
        super(AdaptiveColorShift, self).__init__()
        self.r_ratio = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.g_ratio = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.b_ratio = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.device = device
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        r1, g1, b1 = input[:,0,:,:].unsqueeze(1), input[:,1,:,:].unsqueeze(1), input[:,2,:,:].unsqueeze(1)
        r2, g2, b2 = target[:,0,:,:].unsqueeze(1), target[:,1,:,:].unsqueeze(1), target[:,2,:,:].unsqueeze(1)
        
        # uniform random values
        one = torch.tensor(1).to(self.device)
        self.r = self.sigmoid(self.r_ratio).to(self.device)
        self.g = self.sigmoid(self.g_ratio).to(self.device)
        self.b = self.sigmoid(self.b_ratio).to(self.device)

        r_weight = torch.FloatTensor(1).uniform_(0, 2).to(self.device)
        g_weight = torch.FloatTensor(1).uniform_(0, 2).to(self.device)
        b_weight = torch.FloatTensor(1).uniform_(0, 2).to(self.device)

        r_weight = torch.clip(r_weight, min=one-self.r, max=one+self.r)
        g_weight = torch.clip(g_weight, min=one-self.g, max=one+self.g)
        b_weight = torch.clip(b_weight, min=one-self.b, max=one+self.b)

        output1 = (b_weight*b1+g_weight*g1+r_weight*r1)/(b_weight+g_weight+r_weight)
        output2 = (b_weight*b2+g_weight*g2+r_weight*r2)/(b_weight+g_weight+r_weight)
        return output1, output2


# Learning rate scheduling 
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


# Fourier transform operation
def fft_spectrum(torch_img):

    out = fft.fftn(torch_img, dim=(-2,-1))
    _, _, h, w = out.shape
    center_h, center_w = h // 2, w // 2
    centered_out = torch.zeros_like(out)

    # shift spectrum
    centered_out[..., :center_h, :center_w] = out[..., center_h:, center_w:]
    centered_out[..., :center_h, center_w:] = out[..., center_h:, :center_w]
    centered_out[..., center_h:, :center_w] = out[..., :center_h, center_w:]
    centered_out[..., center_h:, center_w:] = out[..., :center_h, :center_w]
    return centered_out


# measure performance (PSNR and SSIM)
def calc_ssim(im1, im2):
    im1 = im1.data.cpu().detach().numpy().transpose(1, 2, 0)
    im2 = im2.data.cpu().detach().numpy().transpose(1, 2, 0)

    score = structural_similarity(im1, im2, data_range=1, channel_axis=-1)
    return score


def calc_psnr(im1, im2):
    im1 = im1.data.cpu().detach().numpy()
    im1 = im1[0].transpose(1, 2, 0)
    im2 = im2.data.cpu().detach().numpy()
    im2 = im2[0].transpose(1, 2, 0)
    return peak_signal_noise_ratio(im1, im2)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = batch * 255       #   * 0.5  [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).to(device)
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))
    return batch


# def calc_Freq(torch_img, kernel=3):
#     if kernel == 3:
#         sigma = 3
#     elif kernel == 5:
#         sigma = 1.5
#     else:
#         sigma = 1
#     lowFreq = torchgeometry.image.gaussian_blur(
#         torch_img, (kernel, kernel), (sigma, sigma)
#     )
#     highFreq = torch_img - lowFreq
#     highFreq = RGB2gray(highFreq)
#     return lowFreq, highFreq


# below codes are from SSD-GAN
# https://github.com/cyq373/SSD-GAN
def RGB2gray(rgb, keepdim=False):
    if rgb.size(1) == 3:
        r, g, b = rgb[:,0,:,:], rgb[:,1,:,:], rgb[:,2,:,:]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray if not keepdim else gray.unsqueeze(1)
    elif rgb.size(1) == 1:
        return rgb[:,0,:,:] if not keepdim else rgb


# Azimutal Averaging Operation
def azimuthalAverage(image, center=None):
    # Calculate the indices from the image
    H, W = image.shape[0], image.shape[1]
    y, x = np.indices([H, W])
    # radius = np.sqrt((x - H/2)**2 + (y - W/2)**2)  # 每个点半径值（64,192）
    radius = np.sqrt((x - W/2)**2 + (y - H/2)**2)  # 每个点半径值（64,192）
    radius = radius.astype(int).ravel() # 将二维图展平，64*192=12288，有些点的半径值是重复的
    nr = np.bincount(radius) # 统计各个半径出现的频率从0开始：(3,2,1,10,2)
    tbin = np.bincount(radius, image.ravel()) # 12288个频点，和对应的功率谱的乘积的统计
    radial_prof = tbin / (nr + 1e-10) # 计算每个频点的平均功率
    return radial_prof[1:-2]


def shift(x: torch.Tensor):
    out = torch.zeros_like(x)

    H, W = x.size(-2), x.size(-1)
    out[:,:int(H/2),:int(W/2)] = x[:,int(H/2):,int(W/2):]
    out[:,:int(H/2),int(W/2):] = x[:,int(H/2):,:int(W/2)]
    out[:,int(H/2):,:int(W/2)] = x[:,:int(H/2),int(W/2):]
    out[:,int(H/2):,int(W/2):] = x[:,:int(H/2),:int(W/2)]
    return out


def get_fft_feature(x):
    x_rgb = x.detach()
    epsilon = 1e-8

    x_gray = RGB2gray(x_rgb)
    # fft = torch.fft.rfft(x_gray,2,onesided=False)
    fft = torch.view_as_real(torch.fft.fft2(x_gray, dim=(-2,-1)))
    fft += epsilon
    magnitude_spectrum = torch.log((torch.sqrt(fft[:,:,:,0]**2 + fft[:,:,:,1]**2 + 1e-10))+1e-10)
    magnitude_spectrum = shift(magnitude_spectrum)
    magnitude_spectrum = magnitude_spectrum.cpu().numpy()

    out = []
    for i in range(magnitude_spectrum.shape[0]):
        out.append(torch.from_numpy(azimuthalAverage(magnitude_spectrum[i])).float().unsqueeze(0))
    out = torch.cat(out, dim=0)
    
    out = (out - torch.min(out, dim=1, keepdim=True)[0]) / (torch.max(out, dim=1, keepdim=True)[0] - torch.min(out, dim=1, keepdim=True)[0])
    out = Variable(out, requires_grad=True).to(x.device)
    return out


def shift2(x: torch.Tensor):
    out = torch.zeros_like(x)
    H, W = x.size(-2), x.size(-1)
    out[...,:int(H/2),:int(W/2)] = x[...,int(H/2):,int(W/2):]
    out[...,:int(H/2),int(W/2):] = x[...,int(H/2):,:int(W/2)]
    out[...,int(H/2):,:int(W/2)] = x[...,:int(H/2),int(W/2):]
    out[...,int(H/2):,int(W/2):] = x[...,:int(H/2),:int(W/2)]
    return out


def get_fft_feature2(x):
    x_rgb = x.detach()
    epsilon = 1e-8
    fft = torch.view_as_real(torch.fft.fft2(x_rgb, dim=(-2,-1)))
    fft += epsilon
    magnitude_spectrum = torch.log((torch.sqrt(fft[...,0]**2 + fft[...,1]**2 + 1e-10))+1e-10)
    magnitude_spectrum = shift2(magnitude_spectrum)
    return magnitude_spectrum



def split_dir(path, key):

    with_key_path = os.path.join(path, f"{key}")
    without_key_path = os.path.join(path, f"without_{key}")
    if os.path.exists(with_key_path):
        os.shutil.rmtree(with_key_path)
    if os.path.exists(without_key_path):
        os.shutil.rmtree(without_key_path)
    os.makedirs(with_key_path, exist_ok=True)
    os.makedirs(without_key_path, exist_ok=True)

    paths = glob.glob(os.path.join(path, f"*.[jp][pn]g"))
    for path in paths:
        if key in path:
            shutil.move(path, os.path.join(with_key_path, os.path.split(path)[-1]))
        else:
            shutil.move(path, os.path.join(without_key_path, os.path.split(path)[-1]))
        print(f"{os.path.split(path)[-1]} been moved.")
    

class Accumulator:
    "Accmulate on N variables"
    def __init__(self) -> None:
        self.data = {}
    
    def add(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.data:
                self.data[k] += v
            else:
                self.data[k] = v
    
    def reset(self):
        for k in self.data.keys():
            self.data[k] = 0.0
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __getattr__(self, key):
        return self.data[key]
    
    @property
    def postfix(self):
        return { k: f"{(v / self.n_iter):.5}" for idx, (k, v) in enumerate(self.data.items()) if k != 'n_iter'}



if __name__ == "__main__":
    # x = torch.randn(1, 3, 64, 64)
    # print(get_fft_feature(x).shape)
    split_dir(path=r"D:\learning-code\data\denoise\exp\awgn_sigma15\4756\testimages", key='noisy')
