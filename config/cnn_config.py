IMG_SIZE = 28
NUM_CHANNELS = 1
NUM_FEATURES = 1
NUM_CONV = 5
NUM_FC = 3
TEST_SIZE = 0.5
CNN_CONFIG = {
	'conv_kernel_config': {
		'size': 3,
		'depth_init': 32
	},
	'pool_size': 2,
	'dropout_config': {
		'keep_prob_init': 0.9,
		'decay': 0.1
	},
	'fc_relu_config': {
		'size': 1000
	}
}
NODE_NAMES = {
	'inputs': {
		'data': 'input_data',
		'bool': 'input_bool'
	},
	'output': 'output'
}

LEARNING_RATE = 0.001
MOMENTUM = 0.9
DROPOUT = True

EPOCHS = 1001
VAL_EPOCH = 100
SAVE_EPOCH = 200
BATCH_SIZE = 36

LOG_DIR = './cnn_log/'
SAVE_DIR = './cnn_save/'
MODEL_NAME = 'cnn'
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS]
LOSS_TYPE = 'mse'  # ['mse', 'cross_entropy']
OPTIMIZER_TYPE = 'adam'  # ['adam', 'momentum']

DEVICES = {'GPU': 1}
