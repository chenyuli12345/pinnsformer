# implementation of PINNsformer
# paper: PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks
# link: https://arxiv.org/abs/2307.11833

import torch
import torch.nn as nn
import pdb

from util import get_clones

# https://github.com/Zero-coder/FECAM/blob/main/layers/dctnet.py

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import numpy as np
import torch

"""
这段代码实现了一个名为 `dct_channel_block` 的 PyTorch 模块，该模块包含了离散余弦变换（DCT）和通道注意力机制的功能。

首先，定义了一个函数 `dct`，用于对输入的时间序列数据进行离散余弦变换。在函数内部，首先将输入的数据进行处理，然后利用快速傅里叶变换（FFT）相关的操作实现了频域的变换，并最终得到了变换后的频域数据。这个函数将在模块的正向传播过程中被调用，用来对每个通道的数据进行 DCT 变换。

接着，定义了一个名为 `dct_channel_block` 的 PyTorch 模块，它继承自 `nn.Module`。在初始化函数 `__init__` 中，该模块包含了一个神经网络模型，其中通过两个线性层和激活函数构成了一个全连接神经网络。此外，还定义了一个层归一化操作 `dct_norm`，用于对 DCT 变换后的频域数据进行归一化处理。

在前向传播函数 `forward` 中，输入数据 `x` 的形状为 `(B, C, L)`，其中 `B` 表示批次大小，`C` 表示通道数，`L` 表示时间序列的长度。首先对每个通道的数据分别调用之前定义的 `dct` 函数，得到频域数据，并将这些频域数据存储在一个列表中。接着，将列表中的频域数据堆叠起来，得到一个新的张量 `stack_dct`，其形状为 `(B, C, L)`。然后对 `stack_dct` 进行归一化处理，并通过前面定义的全连接神经网络模块 `fc` 对频域数据进行权重调整。最后，将输入数据 `x` 与调整后的权重 `lr_weight` 相乘，得到最终的输出结果。

总之，这个 `dct_channel_block` 模块实现了对输入数据进行离散余弦变换和通道注意力机制的功能，可以被用于深度学习模型中对时间序列数据的处理和特征提取。
"""

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim=(-d))
        r = torch.stack((t.real, t.imag), -1)
        return r


    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:, :, 0], x[:, :, 1]), dim=(-d))
        return t.real


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * 2, bias=False),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2, channel, bias=False),
            nn.Sigmoid()
        )

        self.dct_norm = nn.LayerNorm([96], eps=1e-6)  # for lstm on length-wise

    def forward(self, x):
        b, c, l = x.size()  # (B,C,L)
        list = []
        for i in range(c):
            freq = dct(x[:, i, :])
            # print("freq-shape:",freq.shape)
            list.append(freq)

        stack_dct = torch.stack(list, dim=1)
        stack_dct = torch.tensor(stack_dct)
        lr_weight = self.dct_norm(stack_dct)
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight)

        return x * lr_weight  # result

#自定义激活函数Wavelet：w1 * sin(x) + w2 * cos(x)
class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__() 
        #自定义两个可学习参数w1和w2，初始化为1，训练过程中会更新这两个参数
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x)+ self.w2 * torch.cos(x)

#MLP组件，第一层形状为(d_model, d_ff)，第二层形状为(d_ff, d_ff)，第三层形状为(d_ff, d_model)，前两个使用自定义激活函数WaveAct
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)

#编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True) #多头注意力，设置嵌入维度为d_model，头数为heads，batch_first=True表示输入和输出tensor以(批量大小,序列长度,嵌入维度) 形式提供
        self.ff = FeedForward(d_model) #MLP组件，输入和输出的形状均为d_model，隐藏层的形状为d_ff默认
        self.act1 = WaveAct() #激活函数
        self.act2 = WaveAct() #激活函数
        
    def forward(self, x): #形状为(点的数量，序列长度，d_model)
        x2 = self.act1(x) #对输入x应用激活函数
        # pdb.set_trace()
        x = x + self.attn(x2,x2,x2)[0] #多头注意力机制，查询、键和值均为原始输入经过激活函数后的x2，然后使用残差连接原始输入得到自注意力的输出
        x2 = self.act2(x) #对自注意力的输出应用激活函数
        x = x + self.ff(x2) #将通过激活函数后的自注意力输出通过MLP组件后残差连接原始自注意力输出，得到编码器的输出
        return x 


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True) # 多头注意力，设置嵌入维度为d_model，头数为heads，batch_first=True表示输入和输出tensor以(批量大小,序列长度,嵌入维度) 形式提供
        self.ff = FeedForward(d_model) #MLP组件，输入和输出的形状均为d_model，隐藏层的形状为d_ff默认
        self.act1 = WaveAct() #激活函数
        self.act2 = WaveAct() #激活函数

    def forward(self, x, e_outputs): #输入形状为(点的数量，序列长度，d_model)，e_outputs为编码器的输出，形状也是为(点的数量，序列长度，d_model)
        x2 = self.act1(x) #将解码器输入x通过激活函数
        x = x + self.attn(x2, e_outputs, e_outputs)[0] #多头注意力机制，查询为解码器输入经过激活函数后的x2，键和值为编码器的输出e_outputs，然后使用残差连接原始输入得到自注意力的输出
        x2 = self.act2(x) #将自注意力的输出通过激活函数
        x = x + self.ff(x2) #将通过激活函数后的自注意力输出通过MLP组件后残差连接原始自注意力输出，得到解码器的输出
        return x

#编码器层，包含多个堆叠的编码器架构
class Encoder(nn.Module):
    def __init__(self, d_model, N, heads): #参数N代表编码器的头数，N代表包含几个编码器，d_model是数据的嵌入维度（用于每个编码器），heads是头的数量
        super(Encoder, self).__init__()
        self.N = N 
        self.layers = get_clones(EncoderLayer(d_model, heads), N) #N个编码器
        self.act = WaveAct()

    def forward(self, x): #输入一次通过N个编码器，x的形状为(点的数量，序列长度，d_model)
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x) #编码器的输出要经过激活函数

#解码器层，包含多个堆叠的解码器架构
class Decoder(nn.Module):
    def __init__(self, d_model, N, heads): #参数N代表解码器的头数，N代表包含几个解码器，d_model是数据的嵌入维度（用于每个解码器），heads是头的数量
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N) #N个解码器
        self.act = WaveAct()
        
    def forward(self, x, e_outputs): #输入一次通过N个解码器层，x的形状为(点的数量，序列长度，d_model)，e_outputs为编码器的输出，形状也是为(点的数量，序列长度，d_model)
        for i in range(self.N): #输入和编码器输出依次通过N个解码器
            x = self.layers[i](x, e_outputs) 
        return self.act(x) #解码器的输出要经过激活函数



class PINNsformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads): #接受参数为：d_out代表整个模型的输出的维度（就是最后的mlp层的输出维度），d_model代表每个点的特征嵌入维度，d_hidden代表最后的output layer模块中的隐藏层的维度，N代表编码器和解码器的层数，heads代表多头注意力机制的头数
        super(PINNsformer, self).__init__()

        self.linear_emb = nn.Linear(2, d_model) #线性层，将输入的2维数据映射到d_model维度

        self.dct = dct_channel_block(channel=d_model)

        self.encoder = Encoder(d_model, N, heads) #编码器层
        self.decoder = Decoder(d_model, N, heads) #解码器层

        #输出的mlp模块，包含三个线性层，前两个使用自定义激活函数WaveAct，最终将feature从d_model维度映射到d_hidden维度，最后将d_hidden维度映射到d_out维度
        self.linear_out = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        ])

    def forward(self, x, t): #输入的x和t形状一般为(点的数量, 序列长度, 1)
        src = torch.cat((x,t), dim=-1) #将x和t沿最后一个维度拼接，形成输入的特征向量，形状变为(点的数量，序列长度，2)
        src = self.linear_emb(src) #特征向量通过第一个线性层，映射到高维，此时src的形状为(点的数量，序列长度，d_model)，d_model代表每个点的特征

        src = self.dct(src) #对src进行dct_channel_block处理，得到经过离散余弦变换和通道注意力机制的特征向量，形状仍为(点的数量，序列长度，d_model)

        e_outputs = self.encoder(src) #编码器处理输入特征向量，得到编码后的输出，形状为(点的数量，序列长度，d_model)
        d_output = self.decoder(src, e_outputs) #解码器处理输入特征向量和编码后的输出，得到解码后的输出，形状为(点的数量，序列长度，d_model)
        output = self.linear_out(d_output) #解码后的输出通过输出的mlp模块，得到最终的预测结果。形状为(点的数量，序列长度，d_out)，d_out代表输出的维度
        # pdb.set_trace()
        # raise Exception('stop')
        return output

