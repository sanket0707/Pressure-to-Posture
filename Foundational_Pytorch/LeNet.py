import numpy as np
import torch
import torch.nn as nn


################################################################
#################### Step 1 - Datset ###########################
################################################################
from torchvision import datasets, transforms

def get_data(batch_size, data_root='root', num_worker=1):

    train_test_traonsform = transforms.Compose([
        transforms.Resize(size= ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])


    train_dataloader = torch.utils.data.Dataloader(
        datasets.MNIST(root= , train= , download= False, transform= ),
        batch_size = batch_size,
        shuffle = True,
        num_worker = num_worker
    )


    test_dataloader = torch.utils.data.Dataloader(
        datasets.MNIST(root= , test = , download= False, transform= ),
        batch_size = batch_size,
        shuffle = True,
        num_worker = num_worker
    )

    return train_dataloader, test_dataloader



################################################################
################## Step 2 - LeNet Architecture##################
################################################################
class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._body = nn.Sequential(
            nn.Conv2d(in_channels=1 , out_channels=6 , kernel_size= 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self._head = nn.Sequential(
            nn.Linear(in_features= 16*5*5 , out_features= 120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features= 120 , out_features= 84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features= 10)
        )


    def forward(self,x):
        # Pass the image into main body ie. convolution
        x = self._body(x)
        # Get the output of the main body and the flatten it.
        x = x.view(x.size()[0],-1)
        # pass that flattened output into head
        x = self._head(x)
        # return the output
        return x

LeNet_model = LeNet()
print(LeNet_model)


################################################################
############ Step 3 - System, training Configurations ##########
################################################################
@dataclasses
class systemConfiguration:
    '''
        Describes the common system setting needed for reproducible training
        '''
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)

@dataclasses
class TrainingConfiguration:
    '''
        Describes configuration of the training process
        '''
    batch_size: int = 32  # amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 20  # number of times the whole dataset will be passed through the network
    learning_rate: float = 0.01  # determines the speed of network's weights update
    log_interval: int = 100  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    data_root: str = "data"  # folder to save MNIST data (default: data/mnist-data)
    num_workers: int = 10  # number of concurrent processes used to prepare data
    device: str = 'cuda'  # device to use for training.


def setup_system(system_config: SystemConfiguration) -> None:
    torch.manual_seed(system_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


################################################################
################ Step 4 - Training #######################
################################################################
import torch.nn.functional as F
def train(train_config: TrainingConfiguration,
          model: nn.Module,
          optimizer: torch.optim.optimizer ,
          train_loader: torch.utils.data.Dataloader,
          epoch_idx: int ) -> tuple[float, float]:
    # change model in training mode
    model.train()
    # to get batch loss
    batch_loss = np.array([])
    # to get batch accuracy
    batch_acc = np.array([])
    for batch_idx, (data, target) in enumerate(train_loader):
        # clone target
        indx_target = target.clone()
        # send data to device (it is mandatory if GPU has to be used)
        data = data.to(train_config.device)
        # send target to device
        target = target.to(train_config.device)
        # reset parameters gradient to zero
        optimizer.zero_grad()
        # forward pass to the model
        output = model(data)
        # cross entropy loss
        loss = F.cross_entropy(output,target)
        # find gradients w.r.t training parameters
        loss.backward()
        # Update parameters using gradients
        optimizer.step()
        # get probability score using softmax
        prob = F.softmax(output,dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()
        # accuracy
        acc = float(correct) / float(len(data))
        batch_acc = np.append(batch_acc,[acc])

        if batch_idx % train_config.log_interval == 0 and batch_idx > 0:
            print(
                'Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
                    epoch_idx, batch_idx * len(data), len(train_loader.dataset), loss.item(), acc
                )
            )



    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.mean()
    return epoch_loss, epoch_acc





################################################################
################ Step 5 - Validation  ######################
################################################################
def val(train_config : TrainingConfiguration,
        model : nn.Module,
        test_loader : torch.utils.data.Dataloader
) -> tuple[float, float] :

    model.eval()
    test_loss = 0
    count_correct_prediction = 0

    with torch.not_equal():

        for data, target in test_loader:
            indx_target = data.clone()
            data = data.to(train_config.device)

            target = target.to(train_config.device)
            output = model(data)

            test_loss += F.cross_entropy(output, target).item()
            prob = F.softmax(test_loss, dim=1)

            pred = prob.data.max(dim=1)[1]

            count_correct_prediction = pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / len(test_loader)

        accuracy = 100* count_correct_prediction/len(test_loader.dataset)
        return  test_loss, accuracy/100.0












################################################################
################ Step 6 - Main   ######################
################################################################





################################################################
################ Step 7 - Plot Accuracy   ######################
################################################################






################################################################
############# Step 8 - Save Model Parameters ###################
################################################################


