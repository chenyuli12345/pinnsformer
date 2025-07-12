import numpy as np
import torch.nn as nn
#导入了copy模块。copy模块提供了复制Python对象的功能。
#这个模块提供了浅复制（copy.copy）和深复制（copy.deepcopy）两种复制方式。#浅复制只复制对象本身，而不复制它引用的其他对象；深复制则会复制对象以及它引用的所有对象，并且新对象与原对象完全独立
import copy


#定义一个函数，用于生成一个二维网格数据。接受四个参数，分别代表x和y的范围，以及在这些范围内要生成点的数量。
def get_time_data(t_range, t_num):
    t = np.linspace(t_range[0], t_range[1], t_num) #在t_range[0]和t_range[1]之间生成t_num个点，构成x

    res = t.reshape(-1,1) #将data的形状变为(len(t)*len(x), 2)，即(N, 2)

    t_left = res[0,:].reshape(-1,1)  #取出data的第一行，即左边界，形状为(len(x), 2)

    return res, t_left #返回生成的数据，以及边界数据


#定义一个函数，用于生成一个二维网格数据。接受四个参数，分别代表x和y的范围，以及在这些范围内要生成点的数量。
def get_data(x_range, y_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num) #在x_range[0]和x_range[1]之间生成x_num个点，构成x
    t = np.linspace(y_range[0], y_range[1], y_num) #在y_range[0]和y_range[1]之间生成y_num个点，构成t

    #生成一个二位网络，x_mesh和t_mesh是输出的二维数组，x_mesh的每一行都是x，一共len(t)行，t_mesh的每一列都是t，一共len(x)列，即二者的形状均为(len(t), len(x))
    x_mesh, t_mesh = np.meshgrid(x,t)

    #将x_mesh和t_mesh按照最后一个维度进行拼接，形状为(len(t), len(x), 2)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1) #这里使用了np.expand_dims函数，将x_mesh和t_mesh的维度扩展为原来的维度+1，-1表示在最后一个维度上扩展，即将一维数组扩展为二维数组(例如形状为（3，）变为（3，1）)，将二维数组扩展为三维数组
    
    b_left = data[0,:,:]  #取出data的第一行，即左边界，形状为(len(x), 2)
    b_right = data[-1,:,:] #取出data的最后一行，即右边界，形状为(len(x), 2)
    b_upper = data[:,-1,:] #取出data的最后一列，即上边界，形状为(len(t), 2)
    b_lower = data[:,0,:] #取出data的第一列，即下边界，形状为(len(t), 2)
    res = data.reshape(-1,2) #将data的形状变为(len(t)*len(x), 2)，即(N, 2)

    return res, b_left, b_right, b_upper, b_lower #返回生成的数据，以及四个边界数据

#定义一个函数，用于计算模型的参数数量。传入参数为模型，最后返回模型的参数数量？？？
def get_n_params(model):
    pp=0 #定义一个变量pp，用于存储参数数量
    #遍历模型的所有参数
    for p in list(model.parameters()): #model.parameters()返回模型的所有参数，list将其转换为列表，p代表每一个参数
        nn=1 #定义一个变量nn，用于存储该参数数量
        for s in list(p.size()):  #p.size()是一个元组，包含了参数p在每个维度上的大小，list将其转换为列表，s代表每一个维度的大小
            nn = nn*s #将每个维度的大小相乘，得到该参数的数量。当遍历完p的所有维度后，nn即为该参数的数量
        pp += nn
    return pp

#将输入的二维空间-时间数据src转换为时间序列，为每个空间位置创建一个连续的时间步进序列。输入的数据src形状为(N, 2)，第一列x第二列t。最后返回的是一个伪时间序列数据，形状为(N, num_step, 2)，step则代表Δt，是伪时间序列的递增量。相当于把N个[x,t]转换为{[x,t],[x,t+Δt],[x,t+2Δt],...,[x,t+(num_step-1)Δt]}，即每个空间位置都有一个时间序列。
#最终将形状是(点的数量，2)的数据变为(点的数量，序列长度，2)
def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step 
    #将src的形状从(N, 2)变为(N, L, 2)，首先使用np.expand_dims将src的第二个维度扩展为1，然后使用np.repeat将其沿第二个维度重复dim次，得到(N, L, 2)的形状
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    #遍历每个时间步索引i，然后为t的部分增加一个递增量，相当于是i*Δt
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src

#作用是创建N个module的深度复制，并将它们作为一个nn.ModuleList返回
def get_clones(module, N): #接受两个参数，module是一个pytorch模块，N是一个证书
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#函数主体是一个列表推导式，它会生成一个列表，列表中包含N个module的深度复制。而nn.ModuleList是一个包含各种模块的简单列表，与普通列表相比，其重要特性是会作为模块属性时，会自动注册其包含的模块

def get_data_3d(x_range, y_range, t_range, x_num, y_num, t_num):
    step_x = (x_range[1] - x_range[0]) / float(x_num-1)
    step_y = (y_range[1] - y_range[0]) / float(y_num-1)
    step_t = (t_range[1] - t_range[0]) / float(t_num-1)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]

    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    res = data.reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[0]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_left = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[1]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_right = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[0]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_lower = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[1]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_upper = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    return res, b_left, b_right, b_upper, b_lower