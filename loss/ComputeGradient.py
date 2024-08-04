import torch


def compute_gradients(x, z, matrix):
    n = x.shape[0]
    gradients = []

    for i in range(n):
        # 找到与节点i的邻居index
        connected_indices = matrix[i].nonzero(as_tuple=True)[0]

        # 没有连接的点，则梯度为0
        if len(connected_indices) == 0:
            gradients.append(torch.zeros(3))
            continue

        P_v = x[connected_indices] - x[i]
        S_v = z[connected_indices] - z[i]

        # 计算梯度
        P_v_t = P_v.t()
        try:
            gradient = torch.linalg.solve(P_v_t @ P_v, P_v_t @ S_v)
        except RuntimeError:
            gradient = torch.linalg.pinv(P_v_t @ P_v) @ P_v_t @ S_v

        gradients.append(gradient.squeeze())

    return torch.stack(gradients)


if __name__ == '__main__':
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    matrix = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=torch.float32)
    z = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
