import torch
import torch.nn as nn
import torch.optim as optim

# To get reproducible results use seed
torch.manual_seed(0)

# Define the model
class Diffusion_model(torch.nn.Module):
    def __init__(self,num_input,num_hidden_layer_nodes, num_output ):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(num_input, num_hidden_layer_nodes),
            nn.ReLU(),
            nn.Linear(num_hidden_layer_nodes, num_output)

        )

    def forward(self,X):
        return self.model(X)


num_data = 1000

num_input = 1000
num_hidden_layer_nodes = 100
num_output = 10