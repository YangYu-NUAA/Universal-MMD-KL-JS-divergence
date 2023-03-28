def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算MMD距离，使用RBF核函数
    参数：
        source: 源域样本，形状为(n1, d)
        target: 目标域样本，形状为(n2, d)
        kernel_mul: 核函数中的乘数
        kernel_num: 核函数数量
        fix_sigma: 核函数中的sigma参数
    返回：
        MMD距离值
        
    Calculate MMD distance using RBF kernel function

    Parameters:

    Source: Source domain sample, with a shape of (n1, d)

    Target: Target domain sample, with a shape of (n2, d)

    kernel_ Mul: Multiplier in kernel function

    kernel_ Num: Number of kernel functions

    fix_ Sigma: sigma parameters in kernel functions

    return:

    MMD distance value
    '''
    n1 = source.size()[0]
    n2 = target.size()[0]
    # 将源域和目标域样本拼接在一起
    joint = torch.cat([source, target], dim=0)
    # 计算拼接后样本的范数平方和
    joint_norm = torch.sum(joint * joint, dim=1).reshape(-1, 1)
    # 计算拼接后样本之间的内积矩阵
    joint_dot = torch.matmul(joint, joint.t())
    # 计算拼接后样本之间的欧氏距离平方矩阵（用范数平方和减去内积）
    joint_distance = joint_norm - 2 * joint_dot + joint_norm.t()
    
    # 如果没有指定sigma参数，则根据数据自适应地计算sigma序列
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(joint_distance.data) / (n1 + n2) ** 2

        # 定义RBF核函数（高斯核）
    def guassian_kernel(distance):
        weight = torch.exp(-distance / bandwidth)
        return weight 

     
     # 计算不同核函数下的权重矩阵，并求和得到总权重矩阵
    weights = 0 
    for i in range(kernel_num):
        weights += guassian_kernel(joint_distance * kernel_mul ** i)
     
     # 将总权重矩阵划分为四个部分：S-S、S-T、T-S、T-T 
    s_s = weights[:n1,:n1]
    s_t = weights[:n1,n1:]
    t_s = weights[n1:,:n1]
    t_t = weights[n1:,n1:]
     
    # 根据MMD公式计算最终结果，并返回
    result = torch.mean(s_s) + torch.mean(t_t) - 2 * torch.mean(s_t)
    return result
