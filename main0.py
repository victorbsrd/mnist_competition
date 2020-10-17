import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchsummary import summary
from utils import show_dl, device
from logger import logging
import time

from Project import project
from data import get_dataloaders
from data.Transformation import train_transform, val_transform
from models.cnn import MyCNNClassifier


logging.info(f'Using device = {device}')

# training parameters
params = {
    'lr': 0.001,
    'batch_size': 128,
    'epochs': 2,
    'model': 'cnn'
}

# collect the data
train_dl, val_dl, test_dl = get_dataloaders(
    project.data_dir / "train.csv",
    project.data_dir / "test.csv",
    train_transform = True,
    val_transform = None,
    test_transform = None,
    val_size = 0.2,
    batch_size = params['batch_size']
)

#from utils import show_dl
# check if augmentation is applied
#show_dl(train_dl)
#show_dl(test_dl)

# instanciate model
cnn = MyCNNClassifier(1,10).to(device)
logging.info(summary(cnn, (1,28,28)))

# criterion and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(cnn.parameters(), lr = params['lr'])

writer = SummaryWriter()

def train_model(iter):
    for epochs in range(iter):
        steps = 0
        for x, y in train_dl:
            steps += 1

            #zero grad
            optimer.zero_grad()

            #forward pass
            output = cnn(x)
            p_preds = nn.Sigmoid(output)

            # BCEL loss calculation
            loss = criterion(p_preds, y)

            loss.backward()
            optimizer.step()

            # monitor at some steps
            if(steps//100 = 0)
                writer.add_scalar("Loss/train", loss, steps)

train_model(params['epochs'])
writer.flush()
writer.close()

saving_path = project.savings_dir / f'model={params['model']}_lr={params['lr']}_bs={params['batch_size']}_epochs={param['epochs']}'
torch.save(model.state_dict(), project.savings_dir / )
