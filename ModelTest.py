import torch.optim

from models.GCN import GCN
from utils.Evaluate import plot_loss_vs_epoch
from utils.GraphBuilder import graph_builder

if __name__ == '__main__':
    data = graph_builder()
    print('----------定义图数据为-----------')
    print(data)
    print(data.train_mask)

    model = GCN(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    epochs = []
    losses = []

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        loss = criterion(out[data.train_mask].squeeze(), data.fv[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))

        epochs.append(epoch)
        losses.append(loss.item())

    plot_loss_vs_epoch(losses, epochs)

    with torch.no_grad():
        results = model(data)
        print(results)
        print(results.shape)

        # TODO:写入txt文件中
        result_array = results.numpy()
        with open('./dataset/output.txt', 'w') as file:
            for element in result_array:
                file.write(str(element)[1:-1] + '\n')
