# Reference: https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb

# To  install deps:

# - pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
# Information on this here: https://pytorch.org/get-started/previous-versions/

###############################################################################################################
##################################### Imports from resnet test.py
###############################################################################################################
import torch
import torch.nn as nn
import numpy as np

###############################################################################################################
##################################### Imports from reference
###############################################################################################################
from torch.utils.data import DataLoader
# TODO: May specially need to import torchvision in dependencies
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torchvision.transforms.v2 import ToDtype

###############################################################################################################
##################################### Imports from reference
###############################################################################################################
from torchsummary import summary

###############################################################################################################
##################################### Configs
###############################################################################################################

# For torch
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
SHUFFLE_TRAIN = False                       # for more determinism
device = torch.device('cpu')                # TODO: I don't know if this is necessary because we install CPU only

# For input data
IMAGE_DTYPE = torch.float32
IMAGE_SIZE = [28, 28]

# Model Params
LINEAR_INPUTS = np.prod(IMAGE_SIZE)         # Image dimension TODO: change if resize?
LINEAR_OUTPUTS = 1 #20                      # Will use 20 neurons so will have 20 outputs
OUTPUT_CLASSES = 10                         # Number of classes (10 digits)
USE_BIAS = False                            # TODO: for simplicity at first

###############################################################################################################
##################################### Get dataset (from reference)
###############################################################################################################
batch_size = 1 # Let's just use one for now

data_dir = "./data"
compose = Compose([ToTensor(), ToDtype(IMAGE_DTYPE)])

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=compose)
test_dataset = datasets.MNIST(data_dir, train=False, transform=compose)
# TODO: datasets are originally 28x28 pixels but we should be able to resize smaller, if needed for initial test


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=SHUFFLE_TRAIN) # TODO set shuffle no
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# Each element in data batch looks like: (batch_size, 1, 28, 28)

for (data, target) in train_loader:
    print('data:', data.size(), 'type:', data.type())
    #print(data)
    print('target:', target.size(), 'type:', target.type())
    #print(target)
    input_data_size = data.size()
    break

###############################################################################################################
##################################### Create model (from reference)
###############################################################################################################
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LINEAR_INPUTS, LINEAR_OUTPUTS, bias=USE_BIAS),
            nn.ReLU(),
            nn.Linear(LINEAR_OUTPUTS, OUTPUT_CLASSES, bias=USE_BIAS)
        )

    def forward(self, x):
        return self.layers(x)
model = SimpleMLP().to(device)

###############################################################################################################
##################################### Print model summary (mostly from https://stackoverflow.com/questions/42480111/how-do-i-print-the-model-summary-in-pytorch)
###############################################################################################################
print(model)
summary(model, (28, 28))

###############################################################################################################
##################################### Train the model (from reference): 
###############################################################################################################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def correct(output, target):
    predicted_digits = output.argmax(1)                            # pick digit with largest network output
    correct_ones = (predicted_digits == target).type(torch.float)  # 1.0 for correct, 0.0 for incorrect
    return correct_ones.sum().item()                               # count number of correct ones

def train(data_loader, model, criterion, optimizer):
    model.train()

    num_batches = len(data_loader)
    num_items = len(data_loader.dataset)

    total_loss = 0
    total_correct = 0
    for data, target in data_loader:
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)
        
        # Do a forward pass
        output = model(data)
        
        # Calculate the loss
        loss = criterion(output, target)
        total_loss += loss

        # Count number of correct digits
        total_correct += correct(output, target)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss/num_batches
    accuracy = total_correct/num_items
    print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")

epochs = 1 # TODO: I reduced from 10 to 1
for epoch in range(epochs):
    print(f"Training epoch: {epoch+1}")
    train(train_loader, model, criterion, optimizer)

###############################################################################################################
##################################### Test the model (from reference): 
###############################################################################################################
def test(test_loader, model, criterion):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)

    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)
        
            # Do a forward pass
            output = model(data)
        
            # Calculate the loss
            loss = criterion(output, target)
            test_loss += loss.item()
        
            # Count number of correct digits
            total_correct += correct(output, target)

    test_loss = test_loss/num_batches
    accuracy = total_correct/num_items

    print(f"Testset accuracy: {100*accuracy:>0.1f}%, average loss: {test_loss:>7f}")
test(test_loader, model, criterion)
