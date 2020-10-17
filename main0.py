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
    'model_name': 'cnn'
}

# collect the data
train_dl, val_dl, test_dl = get_dataloaders(
    project.data_dir / "train.csv",
    project.data_dir / "test.csv",
    train_transform = train_transform,
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
model = MyCNNClassifier(1,10).to(device)
logging.info(summary(model, (1,28,28)))

# criterion and optimizer
criterion = torch.nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = params['lr'])


def evaluate_model(model, data_loader):
    model = model.float()
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.float())

            log_softmax = torch.nn.LogSoftmax()
            lp_preds = log_softmax(output)

            loss = criterion(lp_preds, target)
            total_loss += loss.item()

            ps = torch.exp(lp_preds)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            correct_samples+= torch.sum(equals.type(torch.FloatTensor))

    avg_loss = total_loss / total_samples
    avg_acc = correct_samples / total_samples
    print('\nAverage val loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy: ' + '{:4.2f}'.format(100.0 * avg_acc) + '%\n')

    return avg_loss, avg_acc

writer = SummaryWriter()

def train_model(model, iter):
    model = model.float()
    for epochs in range(iter):
        steps = 0
        for x, y in train_dl:
            steps += 1

            #zero grad
            optimizer.zero_grad()

            #forward pass
            output = model(x.float())

            log_softmax= torch.nn.LogSoftmax()
            lp_preds = log_softmax(output)

            # NLL loss calculation
            loss = criterion(lp_preds, y)
            # Backpropagation
            loss.backward()
            optimizer.step()

            # monitor at some steps
            if(steps%100 == 0):
                # monitor the training loss
                writer.add_scalar("Loss/train_batch", loss, steps)
                # monnitor the training Acc
                ps = torch.exp(lp_preds)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == y.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
                writer.add_scalar("Accuracy/train_batch", accuracy, steps)

                print('\nStep: ' + '{}'.format(steps) + ' Batch loss: ' + '{:.4f}'.format(loss) +
                          'Batch accuracy: ' + '{:4.2f}'.format(100.0 * accuracy) + '%\n')

        # at each epochs evaluate the model
        val_loss, val_acc = evaluate_model(model, val_dl)
        writer.add_scalar("Loss/validation", val_loss, steps)
        writer.add_scalar("Acc/validation", val_acc, steps)

train_model(model, params['epochs'])
writer.flush()
writer.close()

saving_path = project.savings_dir / 'model={model_name}_lr={lr}_bs={batch_size}_epochs={epochs}'.format(**params)
torch.save(model.state_dict(), saving_path)
