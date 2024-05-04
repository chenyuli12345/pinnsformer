# baseline implementation of PINNs
# paper: Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
# link: https://www.sciencedirect.com/science/article/pii/S0021999118307125
# code: https://github.com/maziarraissi/PINNs

import torch
import torch.nn as nn

class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):  #用于初始化对象，接受四个参数，分别是输入数据的维度，隐藏层神经元数量，输出数据的维度，隐藏层的数量
        super(PINNs, self).__init__() #调用父类的__init__方法进行初始化

        layers = [] #定义一个空列表，等价于layers = list()
        for i in range(num_layer-1):
            if i == 0:  #对于第一层
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim)) #定义了一个全连接层，输入维度为in_dim，输出维度为hidden_dim（输入层和第一个隐藏层）
                layers.append(nn.Tanh()) 
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)) #定义了全连接层，输入维度为hidden_dim，输出维度为hidden_dim（隐藏层之间）
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim)) #定义了一个全连接层，输入维度为hidden_dim，输出维度为out_dim（最后一个隐藏层和输出层）

        #将所有的层组合成一个Sequential模块，即转换为一个神经网络序列模型，赋值给self.linear
        self.linear = nn.Sequential(*layers)

    #第二个方法forward，用于定义模型的前向传播过程
    def forward(self, x, t): #接受两个参数，分别是输入数据x和t
        src = torch.cat((x,t), dim=-1) #将输入数据x和t在最后一个维度上拼接
        return self.linear(src) #将拼接后的数据输入到self.linear中进行前向传播
    