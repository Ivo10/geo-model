import torch.optim

from loss.ComputeGradient import compute_gradients
from loss.OrientationLoss import OrientationLoss
from models.GCN import GCN
from utils.Evaluate import plot_loss_vs_epoch
from utils.GraphBuilder import graph_builder

if __name__ == '__main__':
    data = graph_builder()
    print('----------定义图数据为-----------')
    print(data)
    print(data.train_mask)

    model = GCN(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion_interface = torch.nn.MSELoss()
    criterion_orientation = OrientationLoss()

    epochs = []
    losses = []

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        gradient = compute_gradients(data.x, data.matrix, out)
        gradient = torch.squeeze(gradient)  # gradient维度应该为n*3

        loss_interface = criterion_interface(out[data.train_mask].squeeze(), data.fv[data.train_mask])
        loss_orientation = criterion_orientation(gradient[data.train_mask], data.alpha[data.train_mask])

        loss = loss_interface + loss_orientation

        loss.backward()
        optimizer.step()
        print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))

        epochs.append(epoch)
        losses.append(loss.item())

    plot_loss_vs_epoch(losses, epochs)

    with torch.no_grad():
        results = model(data)
        print(results)
        print(results.shape)

        result_array = results.numpy()
        with open('./dataset/output.txt', 'w') as file:
            for element in result_array:
                file.write(str(element)[1:-1] + '\n')
