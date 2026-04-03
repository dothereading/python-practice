import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class MultiLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__() #todo: lets look at this later
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim) #todo: why use self.input_dim instead of input_dim
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x.flatten(start_dim=1))
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
def training_loop():
    # load data
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # instantiate NN
    m = MultiLayer(784, 128, 10)

    # training shit: optimizer, loss function
    loss_fn = nn.CrossEntropyLoss()  #todo: what are other kinds of loss, why CE
    optimizer = optim.SGD(params=m.parameters())

    # predict on model
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for image, label in trainloader:
            out = m(image) 
            
            # calculate loss
            loss = loss_fn(out, label)

            # update weights of model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return m

def testing_loop(model):
    # load data
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(testset, batch_size=64, shuffle=True)

    # instantiate NN
    m = model
    m.eval()

    # training shit: optimizer, loss function
    loss_fn = nn.CrossEntropyLoss()  #todo: what are other kinds of loss, why CE

    running_loss = 0.0
    for image, label in trainloader:
        out = m(image) 
        
        # calculate loss
        loss = loss_fn(out, label)

        # update weights of model
        loss.backward()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    m = training_loop()
    testing_loop(m)

    