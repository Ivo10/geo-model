import matplotlib.pyplot as plt

def plot_loss_vs_epoch(losses, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, 'r-', label=u'GCN')
    plt.title('Loss vs. Epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.show()