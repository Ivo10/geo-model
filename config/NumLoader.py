import os


def load_num():
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    with open(project_path + '/dataset/hexahedron.smesh', 'r') as file:
        line = file.readline()
        train_num = list(map(int, line.split()))[0] - 8

    with open(project_path + '/dataset/hexahedron.1.node', 'r') as file:
        line = file.readline()
        node_num = list(map(int, line.split()))[0]

    return train_num, node_num


if __name__ == '__main__':
    train_num, node_num = load_num()
    print(train_num, node_num)
