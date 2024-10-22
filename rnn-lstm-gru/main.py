import torch 
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.dlpack
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt 


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Hyper parameters
# input_size = 784 # 28x28
hiddensize = 128
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = 28
sequence_length = 28
num_layers = 2


# MNIST 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standard global mean and standard deviation from MNIST dataset
])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform)

# Dataloader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

example = next(iter(test_loader))
example_data, example_target = example
print(example_data.shape, example_target.shape)

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(example_data[i][0], cmap='gray')
# plt.show()


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers ,num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


        # x -> (batch_size, seq_len, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # for LSTM


        out, _ = self.rnn(x, h0)
        # out, _ = self.lstm(x, (h0, c0))

        # out: batch_size, seq_length, hidden_size
        # out (N, 28, 128)
        out = out[:, -1, :]
        # out (N, 128)
        out = self.fc(out)
        return out

    

model = RNN(input_size, hiddensize, num_layers, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # original size : 100, 1, 28, 28
        # resized : 100, 28, 28
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass 
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}")

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():

    n_samples = 0
    currect_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        # max returns (value, index)

        # both are same
        # print(torch.max(outputs))
        # print(torch.max(outputs.data))
        _, predictions = torch.max(outputs.data, 1)

        n_samples += len(labels)
        currect_samples += (predictions == labels).sum().item()

acc = 100.0 * (currect_samples / n_samples)
print(f"Accuracy of the network on the 10000 test images: {acc}%")


torch.save(model.state_dict(), "mnist_ffn.pth")