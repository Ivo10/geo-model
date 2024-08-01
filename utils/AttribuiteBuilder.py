import math
import os

import torch

from utils.MaskBuilder import get_train_num


def coordinate_reader(file_path):
    coordinates_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:-1]:  # 去除node文件第一行和最后一行
            coordinate = list(map(float, line.split()))
            coordinates_list.append(coordinate[1:])
    coordinates_tensor = torch.tensor(coordinates_list)

    return coordinates_tensor


def normalize_coordinates(coordinates):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[::, 2]

    x_max, y_max, z_max = torch.max(x), torch.max(y), torch.max(z)
    x_min, y_min, z_min = torch.min(x), torch.min(y), torch.min(z)

    x_c = (x_max + x_min) / 2
    y_c = (y_max + y_min) / 2
    z_c = (z_max + z_min) / 2

    max_range = torch.max(torch.tensor([x_max - x_min, y_max - y_min, z_max - z_min])) / 2

    x_hat = (x - x_c) / max_range
    y_hat = (y - y_c) / max_range
    z_hat = (z - z_c) / max_range

    return torch.stack((x_hat, y_hat, z_hat), dim=1)


def get_fv_by_z(coordinate):
    y, z = coordinate[1], coordinate[2]
    # 构建曲面sin(pai * y / 5)
    bounder = math.sin(math.pi * y / 5)
    if z > (bounder + 4):
        return 0.66
    elif z <= (bounder + 4) and z > (bounder + 3):
        return 0.33
    elif z <= (bounder + 3) and z > (bounder + 2):
        return 0
    elif z <= (bounder + 2) and z > (bounder + 1):
        return -0.33
    else:
        return -0.66


# 为训练节点生成地层f_v
def label_fv__builder():
    fv_list = []
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    train_num, node_num = get_train_num(project_path + '/dataset/hexahedron.smesh',
                                        project_path + '/dataset/hexahedron.1.node')
    # for coordinate in coordinates[8:8 + train_num]:
    #     fv = get_fv_by_z(coordinate)
    #     fv_list.append(fv)
    # TODO: 50为初始的训练节点，每个地层面上有50个
    for i in range(50):
        fv_list.append(-0.33)
    for i in range(50):
        fv_list.append(0)
    for i in range(50):
        fv_list.append(0.33)
    for i in range(50):
        fv_list.append(0.66)

    fv_tensor = torch.cat((torch.zeros(8),
                           torch.tensor(fv_list),
                           torch.zeros(node_num - train_num - 8)))
    return fv_tensor


if __name__ == '__main__':
    print(coordinate_reader('D:\My_Code\python\structure-model\dataset\hexahedron.1.node'))
    # print(normalize_coordinates(coordinate_reader('../dataset/hexahedron.1.node')))
    # print(label_builder())
