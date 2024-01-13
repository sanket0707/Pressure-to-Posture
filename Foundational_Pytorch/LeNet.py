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
def train():
    # change model in training mode

    # to get batch loss
    batch_loss = np.array([])
    # to get batch accuracy
    batch_acc = np.array([])
    for batch_idx, (data, target) in enumerate(train_loader):
        # clone target
        # send data to device (it is mandatory if GPU has to be used)
        # send target to device
        # reset parameters gradient to zero
        # forward pass to the model
        # cross entropy loss
        # find gradients w.r.t training parameters
        # Update parameters using gradients
        # get probability score using softmax
        # get the index of the max probability
        # correct prediction
        # accuracy

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






################################################################
################ Step 6 - Main   ######################
################################################################





################################################################
################ Step 7 - Plot Accuracy   ######################
################################################################






################################################################
############# Step 8 - Save Model Parameters ###################
################################################################



