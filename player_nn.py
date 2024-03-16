# Import necessary PyTorch libraries for neural network construction and operations.
# nn: Contains modules and utilities to build neural networks.
# torch: The main library for tensor operations and automatic differentiation.
import torch
from torch import nn
import numpy as np  # For numerical operations, especially useful for shape manipulations.

class PlayerNN(nn.Module):
    """
    Defines a neural network for the player using PyTorch. This network is specifically designed
    for processing game states in a reinforcement learning context, predicting action values.
    """
    def __init__(self, input_shape, n_actions, freeze=False):
        """
        Initialize the neural network.

        Parameters:
        - input_shape: The shape of the input observations from the environment.
        - n_actions: The number of possible actions the player can take.
        - freeze: If True, the network's weights will not be updated during training.
        """
        super().__init__()  # Initialize the base class (nn.Module).

        # Define the convolutional layers of the network.
        # These layers are designed to process spatial hierarchical features in the input images.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # First convolutional layer.
            nn.ReLU(),  # Activation function to introduce non-linearity.
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Second convolutional layer.
            nn.ReLU(),  # Activation function.
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Third convolutional layer.
            nn.ReLU(),  # Activation function.
        )

        # Calculate the size of the output from the convolutional layers.
        # This is necessary to know how many units the first linear layer needs.
        conv_out_size = self._get_conv_out(input_shape)

        # Define the linear (fully connected) layers of the network.
        self.network = nn.Sequential(
            self.conv_layers,  # The convolutional layers defined above.
            nn.Flatten(),  # Flatten the output of the conv layers for the linear layers.
            nn.Linear(conv_out_size, 512),  # First linear layer.
            nn.ReLU(),  # Activation function.
            nn.Linear(512, n_actions)  # Output layer, one unit per possible action.
        )

        if freeze:
            # If freezing is enabled, make the network's parameters non-trainable.
            self._freeze()
        
        # Determine whether to use GPU (CUDA) or CPU based on availability.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Move the model to the appropriate device.
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass of the network. This method is called to predict the action values.

        Parameters:
        - x: The input state(s) for which action values need to be predicted.
        
        Returns:
        - The predicted action values.
        """
        return self.network(x)

    def _get_conv_out(self, shape):
        """
        Helper method to calculate the output size of the convolutional layers.

        Parameters:
        - shape: The input shape.

        Returns:
        - The total number of units in the output of the conv layers.
        """
        # Pass a dummy input through the convolutional layers to get the output shape.
        o = self.conv_layers(torch.zeros(1, *shape))
        # Calculate the product of dimensions of the output to determine the total number of units.
        return int(np.prod(o.size()))
    
    def _freeze(self):
        """
        Freeze the network's parameters by disabling gradient calculations.
        This makes the model non-trainable.
        """
        for p in self.network.parameters():
            p.requires_grad = False  # Disable gradients for each parameter.