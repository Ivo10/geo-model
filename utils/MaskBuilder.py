import os

import torch


# 从smesh文件中获取训练节点个数，从node文件中获取生成的节点个数
def get_train_num(smesh_path, node_path):
    with open(smesh_path, 'r') as file:
        line = file.readline()
        train_num = list(map(int, line.split()))[0] - 8  # 去除8个顶点

    with open(node_path, 'r') as file:
        line = file.readline()
        node_num = list(map(int, line.split()))[0]

    return train_num, node_num


# 生成train_mask和test_mask
def mask_builder():
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    train_num, node_num = get_train_num(project_path + '/dataset/hexahedron.smesh',
                                        project_path + '/dataset/hexahedron.1.node')

    train_mask = torch.cat([torch.zeros(8),
                            torch.ones(train_num),
                            torch.zeros(node_num - train_num - 8)],
                           dim=0).bool()
    test_mask = torch.cat([torch.ones(8),
                           torch.zeros(train_num),
                           torch.ones(node_num - train_num - 8)],
                          dim=0).bool()

    return train_mask, test_mask


if __name__ == '__main__':
    print(get_train_num('D:\My_Code\python\structure-model\dataset\hexahedron.smesh',
                        'D:\My_Code\python\structure-model\dataset\hexahedron.1.node'))
    train_mask, test_mask = mask_builder()
    # print(train_mask.shape)
    # print(test_mask.shape)
    print(torch.sum(train_mask))
    print(torch.sum(test_mask))
