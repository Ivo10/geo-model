import math

import torch
from utils.AttribuiteBuilder import normalize_coordinates, coordinate_reader
from utils.MaskBuilder import get_train_num


# 为训练节点生成法向量alpha
def label_alpha_builder():
    train_num, node_num = get_train_num('D:\My_Code\python\structure-model\dataset\hexahedron.smesh',
                                        'D:\My_Code\python\structure-model\dataset\hexahedron.1.node')
    normalized_coordinates = normalize_coordinates(coordinate_reader('../dataset/hexahedron.1.node'))
    label_alpha_list = []
    for coordinate in normalized_coordinates[8:258]:
        y = coordinate[1]
        alpha_x = 0
        alpha_y = (math.pi / 5) * math.cos(math.pi * y)
        alpha_z = 1
        label_alpha_list.append((alpha_x, alpha_y, alpha_z))
    label_alpha_tensor = torch.cat((torch.zeros((8, 3)),
                                    torch.tensor(label_alpha_list),
                                    torch.zeros((node_num - train_num - 8, 3))))
    return label_alpha_tensor

if __name__ == '__main__':
    print(label_alpha_builder())