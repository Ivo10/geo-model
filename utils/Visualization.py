import os

import numpy as np
import pyvista as pv

from utils.AttribuiteBuilder import coordinate_reader


# 可视化四面体网格
def visulization():
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    points = coordinate_reader(project_path + '/dataset/hexahedron.1.node').numpy()
    cells = []
    cell_type = []
    with open(project_path + '/dataset/hexahedron.1.ele') as file:
        lines = file.readlines()
        for line in lines[1:-1]:
            numbers = list(map(int, line.split()))[1:]
            numbers = [num - 1 for num in numbers]
            numbers.insert(0, 4)
            cells.append(numbers)
            cell_type.append(10)

    cells = np.array(cells).flatten()
    cell_type = np.array(cell_type).flatten()

    cells = np.hstack([cells])

    # 创建无结构网格
    tetra_mesh = pv.UnstructuredGrid(cells, cell_type, points)

    point_data = []
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    with open(project_path + '/dataset/output.txt') as file:
        lines = file.readlines()
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                float_number = float(cleaned_line)
                point_data.append(float_number)

    tetra_mesh.point_data['values'] = point_data

    plotter = pv.Plotter()
    plotter.add_mesh(tetra_mesh, show_edges=True, scalars='values')
    plotter.show()


if __name__ == '__main__':
    visulization()
