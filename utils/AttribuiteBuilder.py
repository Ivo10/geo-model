import torch

from utils.MaskBuilder import get_train_num


def coordinate_reader(file_path):
    coordinates_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:-1]: # 去除node文件第一行和最后一行
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

# 为训练节点生成地层f_v
def label_builder():
    fv_list = []
    coordinates = coordinate_reader('D:\My_Code\python\structure-model\dataset\hexahedron.1.node')
    train_num, node_num = get_train_num('D:\My_Code\python\structure-model\dataset\hexahedron.smesh',
                                        'D:\My_Code\python\structure-model\dataset\hexahedron.1.node')
    for i in range(train_num):
        z = coordinates[i][2]
        # 见README.md文档
        if z >= 0 and z < 1:
            fv_list.append(-0.66)
        elif z >= 1 and z < 2:
            fv_list.append(-0.33)
        elif z >= 2 and z < 3:
            fv_list.append(0)
        elif z >= 3 and z < 4:
            fv_list.append(0.33)
        else:
            fv_list.append(0.66)

    fv_tensor = torch.cat((torch.tensor(fv_list), torch.zeros(node_num - train_num)))
    return fv_tensor


if __name__ == '__main__':
    print(coordinate_reader('D:\My_Code\python\structure-model\dataset\hexahedron.1.node'))
    # print(normalize_coordinates(coordinate_reader('../dataset/hexahedron.1.node')))
    # print(label_builder())
