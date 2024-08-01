import torch
import torch.nn.functional as F


class OrientationLoss(torch.nn.Module):
    def __init__(self):
        super(OrientationLoss, self).__init__()

    # nv和grad_z_scalar均为n*3维度，
    def forward(self, nv, grad_z_scalar):
        # 向量归一化
        nv_normalized = F.normalize(nv, p=2, dim=1)
        grad_z_scalar_normalized = F.normalize(grad_z_scalar, p=2, dim=1)

        dot_product_result = torch.bmm(torch.unsqueeze(nv_normalized, 1),
                                       torch.unsqueeze(grad_z_scalar_normalized, -1))

        dot_product_result = dot_product_result.squeeze()

        return nv.shape[0] - torch.sum(dot_product_result)


if __name__ == '__main__':
    nv = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])

    grad_z_scalar = torch.tensor([[0.0, 1.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0]])

    orientation_loss_function = OrientationLoss()

    loss = orientation_loss_function(nv, grad_z_scalar)
    print("Loss: ", loss.item())  # 正交向量，夹角均为90度，cos值为1
