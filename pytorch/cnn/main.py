import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.datasets as datasets
import torch.optim as optim

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=32, 
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=4
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )

        self.fc1 = nn.Linear(in_features=5*5*64, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    
    def forward(self, x: torch.Tensor):
        x = self.conv_block(x)
        x = self.fc1(x.flatten(start_dim=1))
        out = self.fc2(F.relu(x))


        return out
    
def training_loop(cnn):

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = cnn.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0
        for idx, (input, label) in enumerate(trainloader):
            pred = cnn(input)
            optimizer.zero_grad()
            loss = loss_fn(pred, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}")
        print(f"Running Loss: {running_loss}")


def test_loop(cnn):
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    with torch.no_grad():
        correct = incorrect = 0
        total = 0
        for input, label in testloader:
            outputs = cnn(input)
            _, predicted = torch.max(outputs, 1)
            total+=label.size(0)

            correct += (predicted==label).sum().item()
    
        print(f" Correct Percentage: {correct/total * 100}%")




if __name__ == "__main__":
    cnn = MyCNN()
    training_loop(cnn) 
    test_loop(cnn)
