import os

from torch_geometric.data import Data

from utils.AttribuiteBuilder import normalize_coordinates, coordinate_reader, label_fv__builder
from utils.EdgeBuilder import edge_builder
from utils.MaskBuilder import mask_builder


def graph_builder():
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    x = normalize_coordinates(coordinate_reader(project_path + '/dataset/hexahedron.1.node'))
    edge_index = edge_builder(project_path + '/dataset/hexahedron.1.ele')
    train_mask, test_mask = mask_builder()
    fv = label_fv__builder()

    data = Data(x=x, edge_index=edge_index, fv=fv, train_mask=train_mask, test_mask=test_mask)

    return data


if __name__ == '__main__':
    print(graph_builder())