random_seed_const = 42
train_val_split_size = 0.75
epochs_default = 20


# Segmentation model
batch_size_train = 8
batch_size_val = 1

num_workers_train = 1
num_workers_val = 1


ENCODER = 'xception'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['birthmark']
ACTIVATION = 'sigmoid' 
DEVICE = "cuda:0"
OPTIMIZER = "sgd"
LR = 0.0001
MOMENTUM = 0.99
WD = 0.9

# Classification model