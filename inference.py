import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.models import inception_v3
from torchvision.transforms import (CenterCrop, Compose,
                                    Resize, ToTensor)

from utils.constants import (CHECKPOINTS, CHECKPOINTS_MODEL, DATA_DIR, L_RATE,
                             MOMENTUM, PATH)


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


def visualize_model(data: DataLoader, model: inception_v3, device: str, num_images=6):
    """Plot test data"""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return model.train(mode=was_training)
    plt.savefig()


def evaluate_model(data, data_size, model, criterion, device):
    """Evaluate model"""

    model.eval()

    total_loss = 0.0
    total_corrects = 0

    with torch.no_grad():
        # Iterate over data
        for inputs, labels in data:
            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds == labels.data)

        loss = total_loss / data_size
        acc = torch.round(total_corrects.double() / data_size, decimals=3)

    print("Loss:", round(loss, 3))
    print("Accuracy: ", acc.item(), "\n")

    return loss, acc


if __name__ == "__main__":
    # Load the model
    model_checkpoint = torch.load(CHECKPOINTS)

    # ResNet
    model_cp = inception_v3(init_weights=True)
    model_cp.aux_logits = False
    num_ftrs = model_cp.fc.in_features
    model_cp.fc = nn.Linear(num_ftrs, 2)

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.

    model_cp.load_state_dict(model_checkpoint)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cp = model_cp.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_cp.parameters(), lr=L_RATE, momentum=MOMENTUM)

    data_transforms = {
        'val': Compose([Resize(256),
                        CenterCrop(299),
                        ToTensor()]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms["val"])
                      for x in ['train', 'val', "test", PATH]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val', "test", PATH]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', "test", PATH]}
    class_names = image_datasets['train'].classes

    # evaluate on all data sets
    print(f"Evaluating the {CHECKPOINTS_MODEL} on:")
    for partition in ["train", "val", "test", PATH]:
        print('\n' + partition)
        evaluate_model(dataloaders[partition], dataset_sizes[partition], model_cp, criterion, device)

    # Visualize some image examples with predictions
    visualize_model(dataloaders[PATH], model_cp, device)
    plt.savefig(f"results/arbitrary_test_imgs.jpg")
