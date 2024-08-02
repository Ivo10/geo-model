import math
import os

import torch

from config.NumLoader import load_num
from utils.AttribuiteBuilder import normalize_coordinates, coordinate_reader


# 为训练节点生成法向量alpha
def label_alpha_builder():
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    train_num, node_num = load_num()
    normalized_coordinates = normalize_coordinates(coordinate_reader(project_path + '/dataset/hexahedron.1.node'))
    label_alpha_list = []
    for coordinate in normalized_coordinates[8:8 + train_num]:
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
