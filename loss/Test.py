import torch

# 假设我们有一个二维Tensor
tensor = torch.tensor([[0, 1, 0, 1],
                       [1, 0, 1, 0],
                       [0, 0, 1, 1]])

# 假设我们想检查第二行（索引为1的行，因为索引从0开始）
row_index = 1

# 选择行
row = tensor[row_index]

# 找到值为1的元素
ones_mask = row == 1

# 获取这些元素的列索引
# 注意：torch.nonzero()返回的是一个二维Tensor，其中每一行都是一个非零元素的索引（行索引和列索引）
# 由于我们只对列索引感兴趣，并且知道行索引，我们可以只取第二列（索引为1的列）
column_indices = torch.nonzero(ones_mask, as_tuple=True)[0]

print(column_indices)