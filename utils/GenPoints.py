import random

# 生成随机输入点
def generate_points(num_points, x_range, y_range, z_range):
    points = []
    for i in range(num_points):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        z = random.uniform(*z_range)
        points.append((x, y, z))
    return points

if __name__ == '__main__':
    num_points = 50
    x_range = (0, 10)
    y_range = (0, 10)
    z_range = (0, 5)
    new_points = generate_points(num_points, x_range, y_range, z_range)

    for id, (x, y, z) in enumerate(new_points, start=1):
        print(f"{id} {x} {y} {z}")

