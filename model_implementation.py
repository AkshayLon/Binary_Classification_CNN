import torch
from torch import nn

class Conv_nn(nn.Module):

    def __init__(self):
        super(Conv_nn, self).__init__()
        # Convolution layer configurations
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=3)

        # Activation and Pooling configurations
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=529, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x): # x.shape = (8,1,200,200)
        # Apply convolutions
        after_conv1 = self.pool(self.relu(self.conv1(x))) # (8,1,200,200) --> (8,1,99,99)
        after_conv2 = self.pool(self.relu(self.conv2(after_conv1))) # (8,1,99,99) --> (8,1,48,48)
        after_conv3 = self.pool(self.relu(self.conv3(after_conv2))) # (8,1,48,48) --> (8,1,23,23)

        # Pass through fully connected layers and give output
        flatten = nn.Flatten() 
        flattened_output = flatten(after_conv3) # (8,1,23,23) --> (8,1,529)
        final_output = self.fc3(self.fc2(self.fc1(flattened_output))) # (8,1,529) --> (8,1)
        return self.sigmoid((final_output-torch.mean(final_output)).mul(1000))
        