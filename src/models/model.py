import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):

    """
    Feed-Foorward Neural Network model for MNIST.
    """

    def __init__(
        self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout,
    ):
        super().__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if x.shape[1] != 784:
            raise ValueError("Expected 2nd input to have shape 784")

        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # Output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
