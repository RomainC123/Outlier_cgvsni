SEED = 0

# Dataset params
NB_IMGS_TRAIN_NI = 1000
NB_IMGS_TRAIN_CG = 500
NB_IMGS_TEST_PERCLASS = 100
NB_CLASSES = 5 # Natural images and 4 different CG algorithms
BATCH_SIZE = 10

# NICE parameters
INPUT_DIM = 4096 # Don't touch that, not an actual parameter
HIDDEN_DIM = 1000
NUM_LAYERS = 4

# Optimizer params
LR_IMG_MAP = 1e-4
LR_FLOW = 1e-3
BETA1 = 0.9
BETA2 = 0.999

# Training params
ID_CG_TRAIN = 1
EPOCHS_IMG_MAP = 10
EPOCHS_FLOW = 100
K = 2
NU = 0.05

LOG_INTERVAL = 1
