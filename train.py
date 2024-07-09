import os
import time
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision.models import alexnet
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip,
                                    RandomResizedCrop, Resize, ToTensor)

from inference import evaluate_model
from utils.constants import (DATA_DIR, L_RATE, MOMENTUM, NB_EPOCHS, OUTPUT,
                             RUN_NAME)
from utils.functions import get_split_ratio, log_params

cudnn.benchmark = True
plt.ion()  # interactive mode


def train_model(model: alexnet,
                criterion: nn.CrossEntropyLoss,
                optimizer: optim.SGD,
                scheduler: lr_scheduler.StepLR,
                num_epochs=25):
    """ Model training script
        model: Pre-trained model weights
        criterion: Loss function type
        optimizer: Algorithm for minimizing cost function
        scheduler: Set the scheduler to vary the learning rate every n epochs
        num_epochs: Number of epochs
        """

    start = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # MlFlow logging
                if phase == "train":
                    mlflow.log_metric("Train Loss", epoch_loss)
                    mlflow.log_metric("Train Acc", epoch_acc)
                else:
                    mlflow.log_metric("Val Loss", epoch_loss)
                    mlflow.log_metric("Val Acc", epoch_acc)

                mlflow.log_metric("Epoch", epoch + 1)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        training_duration = time.time() - start
        print(f'Training complete in {training_duration // 60:.0f}m {training_duration % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        mlflow.log_metric("Best Val Accuracy", best_acc)

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


if __name__ == "__main__":
    # Provide the path to images
    data_dir = DATA_DIR

    data_transforms = {
        'train': Compose([RandomResizedCrop(224),
                          RandomHorizontalFlip(),
                          ToTensor()]),
        'val': Compose([Resize(256),
                        CenterCrop(224),
                        ToTensor()]),
        'test': Compose([Resize(256),
                         CenterCrop(224),
                         ToTensor()]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    dataset_split_ratio = get_split_ratio(dataset_sizes)
    class_names = image_datasets['train'].classes

    model_ft = alexnet(weights='IMAGENET1K_V1')
    model_ft.aux_logits = False

    num_ftrs = model_ft.classifier[-1].in_features

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=L_RATE, momentum=MOMENTUM)

    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        # ADD LOG PARAMS
        log_params(NB_EPOCHS, DATA_DIR, L_RATE, MOMENTUM,
                   data_transforms, dataset_split_ratio,
                   criterion, optimizer_ft, device)

        # Fine-tuning
        model_ft_output = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                      num_epochs=NB_EPOCHS)

        torch.save(model_ft_output.state_dict(), OUTPUT)

        # Evaluate on test dataset
        _, test_acc = evaluate_model(dataloaders["test"], dataset_sizes["test"], model_ft_output, criterion, device)

        mlflow.log_metric("Test Accuracy", test_acc)

# 178 x 218 pix
