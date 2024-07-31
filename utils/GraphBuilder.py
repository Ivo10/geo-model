from torch_geometric.data import Data

from utils.AttribuiteBuilder import normalize_coordinates, coordinate_reader, label_fv__builder
from utils.EdgeBuilder import edge_builder
from utils.MaskBuilder import mask_builder


def graph_builder():
    x = normalize_coordinates(coordinate_reader('D:\My_Code\python\structure-model\dataset\hexahedron.1.node'))
    edge_index = edge_builder('D:\My_Code\python\structure-model\dataset\hexahedron.1.ele')
    train_mask, test_mask = mask_builder()
    fv = label_fv__builder()

    data = Data(x=x, edge_index=edge_index, fv=fv, train_mask=train_mask, test_mask=test_mask)

    return data


if __name__ == '__main__':
    graph_builder()
