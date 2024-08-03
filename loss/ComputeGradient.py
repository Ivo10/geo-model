import torch


# 定义函数来计算梯度
def compute_gradients(x, z, matrix):
    n = x.shape[0]
    gradients = []

    for i in range(n):
        # 找到与i点相连的点
        connected_indices = matrix[i].nonzero(as_tuple=True)[0]

        # 没有连接的点，则梯度为0
        if len(connected_indices) == 0:
            gradients.append(torch.zeros(3))
            continue

        # 构建P_v和S_v
        P_v = x[connected_indices] - x[i]
        S_v = z[connected_indices] - z[i]

        # 计算梯度
        P_v_t = P_v.t()
        try:
            # 求解正规方程
            gradient = torch.linalg.solve(P_v_t @ P_v, P_v_t @ S_v)
        except RuntimeError:
            # 如果P_v^T P_v是奇异的，使用伪逆
            gradient = torch.linalg.pinv(P_v_t @ P_v) @ P_v_t @ S_v

        gradients.append(gradient.squeeze())

    return torch.stack(gradients)


if __name__ == '__main__':
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    matrix = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    z = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)

    print(compute_gradients(x, z, matrix).shape)
