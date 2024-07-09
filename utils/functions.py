import glob

import mlflow
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.sgd import SGD


def log_params(NB_EPOCHS: int, DATA_DIR: str, L_RATE: float,
               MOMENTUM: float, data_transforms: dict, data_split: str,
               criterion: CrossEntropyLoss,
               optimizer_ft: SGD,
               device: str) -> None:
    """
    Log parameters to MlFlow
    """

    tr_set = len(glob.glob(DATA_DIR + "/train/*/*.jpg"))
    val_set = len(glob.glob(DATA_DIR + "/val/*/*.jpg"))

    mlflow.log_params({"Num epochs": NB_EPOCHS,
                       "Learning Rate": L_RATE,
                       "Momentum": MOMENTUM,
                       "Training set size": tr_set,
                       "Validation set size": val_set,
                       "Data split ratio": data_split,
                       "Data Transformations": data_transforms,
                       "Model Criterion": criterion,
                       "Model Optimizer": optimizer_ft,
                       "Device": device
                       }
                      )


def get_split_ratio(dataset_sizes: list) -> str:
    """Get data split ratio"""

    res = [dataset_sizes[i] / sum([k for i, k in dataset_sizes.items()]) * 100
           for i in ["train", "val", "test"]]

    return f"{round(res[0], 2)}% /{round(res[1], 2)}% /{round(res[2], 2)}%"
