"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
from guided_diffusion.data import * 
import torch as th
import torch.nn.functional as F 
import torch 
from torch import nn
# from guided_diffusion.script_util import mapping


def mapping(tensors,min,max):
    leng = []
    for i in range(tensors.shape[0]):
        leng.append ( tensors[i].max()-tensors[i].min())
        tensor = (tensors[i]-tensors[i].min())/leng[i]*(max-min)+min
        tensors[i] = tensor
    return tensors   



def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        # 输入的图像通道数为3，输出通道数也为3
        self.in_channels = 3
        self.out_channels = 3

        # 定义网络的层
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):


        # 进行前向传播
        out = self.conv1(x+t.mean()/1000)
        out = self.relu(out)

        return out
    
class Align_Loss(nn.Module):
    """
    DCG Loss implementation by Jerry. (2022.11.11)
    """
    def __init__(self, puge_model,classifier1,classifier2):
        super().__init__()
        # 
        self.puge_model = puge_model
        self.classifier1 = classifier1
        self.classifier2 =classifier2

    def _set_up(self, x,t, y1,y2):
        """
        Set up variables.
        """
        self.x = x
        self.t = t

        self.y1 = y1.to(th.int64)
        self.y2 = y2.to(th.int64)
        self.x.requires_grad = True
        self.t.requires_grad = True

        torch.nn.Conv2d

    def get_loss(self):
        """
        Get loss.
        """

        self.adorn_x = self.puge_model.module[1](self.puge_model.module[0](self.x,self.t))
        
        
        self.logits1 = self.classifier1(self.adorn_x)
        self.logits1_y = torch.log(F.softmax(self.logits1,dim=-1).gather(1,self.y1.unsqueeze(1))).mean() # 对数条件概率  越小越好

        self.logits2 = self.classifier2(self.adorn_x) 
        self.logits2_y = torch.log(F.softmax(self.logits2,dim=-1).gather(1,self.y2.unsqueeze(1))).mean() # 对数条件概率 越小越好


        # self.logits1_y / bacth * -1 即是loss1
        
        self.loss1 = F.cross_entropy(self.logits1,self.y1) # 正常的交叉熵损失函数
        self.loss2 = F.cross_entropy(self.logits2,self.y2)
        

        # self.grad1_x,self.grad1_t = torch.autograd.grad(self.loss1,(self.x,self.t),torch.ones_like(self.loss1), retain_graph=True, create_graph=True)
        # self.grad2_x,self.grad2_t = torch.autograd.grad(self.loss2,(self.x,self.t),torch.ones_like(self.loss2), retain_graph=True, create_graph=True)


        self.grad1_x,self.grad1_t = torch.autograd.grad(self.loss1,(self.x,self.t),torch.ones_like(self.loss1), retain_graph=True, create_graph=True)
        self.grad2_x,self.grad2_t = torch.autograd.grad(self.loss2,(self.x,self.t),torch.ones_like(self.loss2), retain_graph=True, create_graph=True)

       
        # self.grad2_g_x,self.grad2_g_t = torch.autograd.grad((self.grad2_x.sum(),self.grad2_t.sum()),(self.x,self.t),(torch.ones_like(self.grad2_x.sum()),torch.ones_like(self.grad2_t.sum())), retain_graph=True, create_graph=True)
        # self.grad1_g_x,self.grad1_g_t = torch.autograd.grad((self.grad1_x.sum(),self.grad1_t.sum()),(self.x,self.t),(torch.ones_like(self.grad1_x.sum()),torch.ones_like(self.grad1_t.sum())), retain_graph=True, create_graph=True)



        self.loss3 = torch.norm(self.grad1_x-self.grad2_x,p=2) + torch.norm(self.grad1_t-self.grad2_t,p=2)
        self.loss3 = self.loss3 * 20

        self.loss = self.loss3  + self.loss1 +  self.loss2   
        return self.loss,self.loss1,self.loss2,self.loss3

    def forward(self, x, y):
        self._set_up(x, y)
        return self.get_loss()