import torch

from config.NumLoader import load_num


# 生成train_mask和test_mask
def mask_builder():
    train_num, node_num = load_num()
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
    train_mask, test_mask = mask_builder()
    # print(train_mask.shape)
    # print(test_mask.shape)
    print(torch.sum(train_mask))
    print(torch.sum(test_mask))
