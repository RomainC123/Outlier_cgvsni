SEED = 0

# Dataset params
NB_IMGS_TRAIN = 1000
NB_IMGS_TEST_PERCLASS = 100
NB_CLASSES = 2 # One for natural images, and then 4 different sources of CG images
BATCH_SIZE = 10

# NICE parameters
INPUT_DIM = 6400 # Don't touch that, not an actual parameter
HIDDEN_DIM = 1000
NUM_LAYERS = 4

# Optimizer params
LR = 10e-4
BETA1 = 0.9
BETA2 = 0.999

# Training params
EPOCHS = 2
K = 2
NU = 0.1

LOG_INTERVAL = 1
