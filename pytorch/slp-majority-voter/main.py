import torch
from torch import nn
from itertools import product
import torch.utils.data as data
import matplotlib.pyplot as plt


class MajorityFunction(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.perceptron = nn.Linear(
            in_features=self.num_inputs, out_features=1, bias=True
        )

    def forward(self, x):
        return self.perceptron(x)


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X).squeeze(1)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch == 0:
            loss = loss.item()
        return loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).squeeze(1)
            test_loss += loss_fn(pred, y).item()
            predicted = (pred > 0).float()
            correct += (predicted == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return test_loss, correct


def make_dataset(n: int):
    permutations = list(product([0.0, 1.0], repeat=n))
    X_train = torch.tensor([list(p) for p in permutations])
    y_train = torch.tensor([1.0 if sum(p) > n / 2 else 0.0 for p in permutations])
    return X_train, y_train


def get_test_loss(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).squeeze(1)
            total_loss += loss_fn(pred, y).item()
    return total_loss / len(dataloader)


def plot_training(
    epochs, train_losses, test_losses, accuracies, weight_history, bias_history, n, weight_decay, batch_size
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_losses, label="Train")
    axes[0, 0].plot(epochs, test_losses, label="Test")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"Loss over time (n={n}, wd={weight_decay})")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [a * 100 for a in accuracies], color="green")
    axes[0, 1].axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title(f"Test accuracy (n={n}, batch_size={batch_size})")
    axes[0, 1].set_ylim(0, 105)

    weight_history = list(zip(*weight_history))
    for i, w in enumerate(weight_history):
        axes[1, 0].plot(epochs, w, label=f"w{i}")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Weights over time")
    axes[1, 0].axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label=f"ideal = 1.0")
    axes[1, 0].legend(fontsize="small", ncol=2)

    axes[1, 1].plot(epochs, bias_history, color="brown", linestyle="--")
    axes[1, 1].axhline(y=-n / 2, color="gray", linestyle=":", alpha=0.5, label=f"ideal = {-n/2}")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].set_title("Bias over time")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"training_plots_n_{n}_weight_decay_{weight_decay}.png", dpi=150)


if __name__ == "__main__":
    # Set n for input size
    # As we vary n, we'll have to adjust the weight_decay if we want the weights and bias to
    # converge to the ideal values described in Russell (p 731) of w_i = 1 for all i and b = -n/2.

    # I tried a few experiments:
    #     For n=5, weight_decay=0.02 seems to work well; b converges to -2.3
    #     For n=10, weight_decay=0.005; b converges to -5.6
    #     For n=20, weight_decay=0.0015; b is at -10.2 but hasn't really converged yet after 10k epochs
    n = 10
    weight_decay = 0.005

    batch_size = 15
    model = MajorityFunction(n)
    X_train, y_train = make_dataset(n)
    trainset = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    # when n gets really big, test set takes way to long unless we limit it
    testloader = data.DataLoader(
        data.Subset(trainset, torch.randperm(len(trainset))[:1000])
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=weight_decay)

    epochs_log, train_losses, test_losses, accuracies = [], [], [], []
    weight_history, bias_history = [], []
    for t in range(5001):
        loss = train_loop(trainloader, model, loss_fn, optimizer, batch_size)

        if t % 25 == 0:
            test_loss, test_acc = test_loop(testloader, model, loss_fn)
            epochs_log.append(t)
            train_losses.append(loss)
            test_losses.append(test_loss)
            accuracies.append(test_acc)
            weight_history.append(model.perceptron.weight[0].detach().tolist())
            bias_history.append(model.perceptron.bias[0].detach().item())

        if t in {
            50,
            100,
            200,
            400,
            750,
            1000,
            1500,
            2000,
            2500,
            3000,
            3500,
            5000,
            7500,
            10000,
        }:
            print(f"Epoch {t + 1}\n-------------------------------")
            print(f"Training Loss: {loss}")
            test_loss, test_correct = test_loop(testloader, model, loss_fn)
            print(
                f"Test Error: \n Accuracy: {(100 * test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
            )

    print(model.perceptron.weight)
    print(model.perceptron.bias)
    plot_training(
        epochs_log,
        train_losses,
        test_losses,
        accuracies,
        weight_history,
        bias_history,
        n,
        weight_decay,
        batch_size,
    )