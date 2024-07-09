######################
# TRAINING PARAMETERS
######################

NB_EPOCHS = 35
L_RATE = 0.001
MOMENTUM = 0.9
SEED = None
RUN_NAME = "Gender Classification: ResNet - 35ep"
OUTPUT = "resnet18_model.pt"

DATA_DIR = 'data/input'

#########################
# INFERENCE PARAMETERS
########################

# Path to model checkpoints
CHECKPOINTS_FOLDER = "checkpoints/"
CHECKPOINTS_MODEL = "resnet18_model.pt"
CHECKPOINTS = CHECKPOINTS_FOLDER + CHECKPOINTS_MODEL

# Path to arbitrary data for inference
PATH = "test_arbitrary"
# Folder to store results
RESULTS = f'{CHECKPOINTS_MODEL.split(".")[0]}_{PATH.split("/")[-1]}'
