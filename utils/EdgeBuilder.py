import os

import torch
import numpy as np

from config.NumLoader import load_num


# 以二元组和邻接矩阵形式，返回图的拓扑结构
def edge_builder(file_path):
    train_num, node_num = load_num()
    adj_matrix = torch.zeros(node_num, node_num)
    edge_set = set()

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:-1]:  # 去除第一行和最后一行
            numbers = list(map(int, line.split()))
            indexes = sorted(numbers[1:])  # set+排序去重
            for i in range(len(indexes)):
                for j in range(i + 1, len(indexes)):
                    # TetGen生成的顶点索引从1开始，而PyG中Data数据类型中x索引从0开始
                    edge_set.add((indexes[i] - 1, indexes[j] - 1))
                    edge_set.add((indexes[j] - 1, indexes[i] - 1))

                    adj_matrix[indexes[i] - 1, indexes[j] - 1] = 1
                    adj_matrix[indexes[j] - 1, indexes[i] - 1] = 1

    # set转tensor
    edges_list = [list(edge) for edge in edge_set]
    edges_numpy = np.array(edges_list)
    edges_tensor = torch.tensor(edges_numpy).T

    return edges_tensor, adj_matrix


if __name__ == '__main__':
    file_path = '../dataset/hexahedron.1.ele'
    edges, matrix = edge_builder(file_path)
    print(edges)
    print(edges.shape)
    print(matrix)
    print(matrix.shape)
    print(torch.sum(matrix))
