import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.archs.CBAM_arch import *


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


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResidualBlock_noBN_CA(nn.Module):
    ''' Residual block w/o BN + channel attention
    ---Conv-ReLU-Conv--——————————————----------|-------|
    |              |--Average pool-FC-ReLU-FC--x------ concat-Conv-
    |------------------------------------------------- |
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_CA, self).__init__()
        reduction_ratio = 16
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(3*nf, nf, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(nf, nf // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction_ratio, nf, bias=False),
            nn.Sigmoid()
        )

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        initialize_weights([self.fc])

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        identity_out = out
        B, C, _, _ = out.size()
        att = self.fc(self.avg_pool(out).view(B, C)).view(B, C, 1, 1)
        out = out * att.expand_as(out)
        return self.conv3(torch.cat([identity, identity_out, out], dim=1))


# add NonLocalBlock2D
# reference: https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        # for pytorch 0.3.1
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        # initialization
        # initialize_weights([self.g, self.W, self.theta, self.phi])


    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0,2,1).contiguous()

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0,2,1).contiguous()

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NonLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        # initialization
        # initialize_weights([self.g, self.W, self.theta, self.phi])

    def forward(self, x):

        B, C, H, W = x.size()
        x = x.view(B, C, -1)  #[B, C, H, W] -> [B, C, HxW]
        x = x.permute(0, 2, 1).contiguous() #[B, C, HxW] -> [B, HxW, C]


        g_x = self.g(x).view(B, self.inter_channels, -1)

        g_x = g_x.permute(0,2,1)

        theta_x = self.theta(x).view(B, self.inter_channels, -1)

        theta_x = theta_x.permute(0,2,1)

        phi_x = self.phi(x).view(B, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0,2,1).contiguous()

        y = y.view(B, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
