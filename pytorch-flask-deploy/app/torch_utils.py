import torch 
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import io


# load model 
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        # no activation and no softmax at the end
        return x

input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)

PATH = "mnist_ffn.pth"
model.load_state_dict(torch.load(PATH))
model.eval()

# image -> tensor 
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Standard global mean and standard deviation from MNIST dataset
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    # max returns (value, index)
    _, predicted = torch.max(outputs, 1)
    return predicted
