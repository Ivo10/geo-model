import os

from torch_geometric.data import Data

from loss.LabelBuilder import label_alpha_builder
from utils.AttribuiteBuilder import normalize_coordinates, coordinate_reader, label_fv_builder
from utils.EdgeBuilder import edge_builder
from utils.MaskBuilder import mask_builder


def graph_builder():
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    x = normalize_coordinates(coordinate_reader(project_path + '/dataset/hexahedron.1.node'))
    edge_index, matrix = edge_builder(project_path + '/dataset/hexahedron.1.ele')
    train_mask, test_mask = mask_builder()

    fv = label_fv_builder()
    alpha = label_alpha_builder()

    data = Data(x=x, edge_index=edge_index, matrix=matrix, fv=fv, alpha=alpha,
                train_mask=train_mask, test_mask=test_mask)

    return data


if __name__ == '__main__':
    print(graph_builder())