NUM_CLASSES = 50
IMAGE_SIZE = 64
BATCH_SIZE = 128

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1600
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 400

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 200.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.0005 # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
