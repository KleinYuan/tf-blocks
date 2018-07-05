INPUT_SIZE = 28

NUM_FEATURES = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.9

NUM_HIDDEN = 64

EPOCHS = 1001
VAL_EPOCH = 100
SAVE_EPOCH = 200
BATCH_SIZE = 36

LOG_DIR = './log/'
SAVE_DIR = './save/'
MODEL_NAME = 'template'
INPUT_SHAPE = [None, INPUT_SIZE, INPUT_SIZE]
LOSS_TYPE = 'mse'  # ['mse', 'cross_entropy']
OPTIMIZER_TYPE = 'adam'  # ['adam', 'momentum']

DEVICES = {'GPU': 1}

NODE_NAMES = {
	'inputs': {
		'data': 'input_data',
		'bool': 'input_bool'
	},
	'output': 'output'
}