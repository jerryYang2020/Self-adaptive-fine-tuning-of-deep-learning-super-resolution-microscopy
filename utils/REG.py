import torch
import torch.nn as nn
import numpy as np

class SparseReg(nn.Module):
    def __init__(self,sparsity = 0.1):
        super(SparseReg,self).__init__()
        self.sparsity = sparsity

    def forward(self,img):
        l1 = torch.norm(img, p=1)
        l1=l1/(img.shape[1]*img.shape[2]*img.shape[3])
        sparse_reg=self.sparsity*l1
        return sparse_reg


class HessianReg(nn.Module):
    def __init__(self,contiz = 0.5):
        super(HessianReg,self).__init__()
        self.contiz = np.sqrt(contiz)

    def forward(self,img):
        delta_xx = torch.tensor([[[[[1, -2, 1]]]]], dtype=torch.float32, device=torch.device('cuda:0'))
        g_xx = torch.norm(nn.functional.conv3d(img, delta_xx, stride=1, padding=0), p=1)
        delta_xy = torch.tensor([[[[[1, -1], [-1, 1]]]]], dtype=torch.float32, device=torch.device('cuda:0'))
        g_xy = torch.norm(nn.functional.conv3d(img, delta_xy, stride=1, padding=0), p=1)
        delta_xz = torch.tensor([[[[[1, -1]], [[-1, 1]]]]], dtype=torch.float32, device=torch.device('cuda:0'))
        g_xz = torch.norm(nn.functional.conv3d(img, delta_xz, stride=1, padding=0), p=1)
        delta_yy = torch.tensor([[[[[1], [-2], [1]]]]], dtype=torch.float32, device=torch.device('cuda:0'))
        g_yy = torch.norm(nn.functional.conv3d(img, delta_yy, stride=1, padding=0), p=1)
        delta_yz = torch.tensor([[[[[1], [-1]], [[-1], [1]]]]], dtype=torch.float32, device=torch.device('cuda:0'))
        g_yz = torch.norm(nn.functional.conv3d(img, delta_yz, stride=1, padding=0), p=1)
        delta_zz = torch.tensor([[[[[1]], [[-2]], [[1]]]]], dtype=torch.float32, device=torch.device('cuda:0'))
        g_zz = torch.norm(nn.functional.conv3d(img, delta_zz, stride=1, padding=0), p=1)
        hessian_reg = g_xx + g_yy + (self.contiz ** 2) * g_zz + 2 * g_xy + 2 * (self.contiz) * g_xz + 2 * (self.contiz) * g_yz
        hessian_reg = hessian_reg/(img.shape[3]*img.shape[4])

        return hessian_reg


class TVReg(nn.Module):
    def __init__(self,t_only=False):
        super(TVReg,self).__init__()
        self.t_only = t_only

    def forward(self,x):
        if self.t_only:
            d_diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
            d_tv = torch.pow(d_diff, 2).mean()
            return torch.sqrt(d_tv)
        else:
            h_diff = x[:, :, :, :, 1:] - x[:, :, :, :, :- 1]
            w_diff = x[:, :, :, 1:, :] - x[:, :, :, :- 1, :]
            d_diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
            h_tv = torch.pow(h_diff, 2).mean()
            w_tv = torch.pow(w_diff, 2).mean()
            d_tv = torch.pow(d_diff, 2).mean()
            return torch.sqrt(h_tv + w_tv + d_tv)