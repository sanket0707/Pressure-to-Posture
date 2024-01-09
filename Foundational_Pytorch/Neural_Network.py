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

num_epochs = 100

#Create input and output tensor

x = torch.rand(num_data, num_input)
y = torch.rand(num_data, num_output)

# construct model
model = Diffusion_model(num_input, num_hidden_layer_nodes, num_output)

# Define Loss Function
loss_function = nn.MSELoss(reduction = 'sum')

# Define optimiser
optimiser = optim.SGD(model.parameters(), lr=1e-4)


for t in range(num_epochs):
    y_pred = model(x)

    loss = loss_function(y_pred, y)
    print(t , loss)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()



