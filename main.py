import warnings
warnings.filterwarnings("ignore")

from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import project
from data import get_dataloaders
from data.Transformation import train_transform, val_transform
from models.cnn import MyCNNClassifier
from utils import show_dl, device
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback
from logger import logging
import time

params = {
    'lr': 0.001,
    'batch_size': 64,
    'model': 'cnn'
}

logging.info(f'Using device = {device}')

# collect the data
train_dl, val_dl, test_dl = get_dataloaders(
    project.data_dir / "train.csv",
    project.data_dir / "test.csv",
    train_transform = None,
    val_transform = None,
    test_transform = None,
    val_size = 0.2,
    batch_size = params['batch_size']
)

# check if augmentation is applied
#show_dl(train_dl)
#show_dl(test_dl)

# Create an experiment
experiment = Experiment(api_key="emXVyWUp9s2OAwC0Y7A6PMLrC",
                        project_name="general", workspace="victorbsrd")
experiment.log_parameters(params)

cnn = MyCNNClassifier(1,10).to(device)
logging.info(summary(cnn, (1,28,28)))

optimizer = optim.Adam(cnn.parameters(), lr = params['lr'])
model = Model(cnn, optimizer, 'cross_entropy', batch_metrics = ['accuracy']).to(device)

callbacks = [ ReduceLROnPlateau(monitor='val_acc', patience= 5, verbose = True),
    ModelCheckpoint(str(project.checkpoint_dir / f'{time.time()}-model.pt'), save_best_only = True, verbose = True),
    EarlyStopping( monitor = 'val_acc', patience = 10, mode = 'max'),
    CometCallback(experiment)
]

model.fit_generator(train_dl,
    val_dl,
    epochs = 10,
    callbacks=callbacks
)

loss, test_acc = model.evaluate_generator(test_dl)
logging.info(f'test_acc=({test_acc})')
experiment.log_metric('test_acc', test_acc)
