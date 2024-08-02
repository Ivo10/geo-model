import torch


def compute_gradients(x, matrix, z):
    """
    生成所有顶点的梯度值

    :param x: 顶点坐标集合(n*3)
    :param matrix: 邻接矩阵(n*n)
    :param z: 隐式场(n*1)
    :return: 所有顶点梯度(n*3)
    """
    gradients_list = []

    for i in range(x.shape[0]):
        gradient = compute_x_gradient(i, x, matrix, z)
        gradients_list.append(gradient)

    gradients = torch.stack(gradients_list,dim=0)
    return gradients


def compute_x_gradient(index, x, matrix, z):
    """
    计算第index个顶点的梯度值

    :param index: 顶点索引
    :param x: 坐标集合(n*3)
    :param matrix: 邻接矩阵(n*n)
    :param z: 隐式场(n*1)
    :return: 第index个顶点梯度(3*1)
    """
    p_v = compute_pv(index, x, matrix)
    s_v = compute_sv(index, z, matrix)

    p_v_p = torch.mm(p_v.T, p_v)

    gradient = torch.mm(torch.mm(p_v_p.inverse(), p_v.T), s_v)
    return gradient


def compute_pv(index, x, matrix):
    """
    计算第index顶点的P矩阵

    :param index:
    :param x:(n*3)
    :param matrix:(n*n)
    :return:第index顶点的P矩阵(m*3)，m为第index节点的度
    """
    x_v = x[index]

    # 找出index节点的所有邻居坐标
    row = matrix[index]

    ones_mask = (row == 1)
    neighbor_indices = torch.nonzero(ones_mask, as_tuple=True)[0]

    neighbors_x = []
    for neighbor_index in neighbor_indices:
        neighbor_x = x[neighbor_index]
        neighbors_x.append(neighbor_x)

    neighbors_x = torch.stack(neighbors_x, dim=0)

    pv = neighbors_x - x_v
    return pv


def compute_sv(index, z, matrix):
    z_v = z[index]

    # 找出index节点的所有邻居坐标
    row = matrix[index]

    ones_mask = (row == 1)
    neighbor_indices = torch.nonzero(ones_mask, as_tuple=True)[0]

    neighbors_z = []
    for neighbor_index in neighbor_indices:
        neighbor_z = z[neighbor_index]
        neighbors_z.append(neighbor_z)

    neighbors_z = torch.stack(neighbors_z, dim=0)

    return neighbors_z - z_v
