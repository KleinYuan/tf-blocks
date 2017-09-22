NODE_NAMES = {
    'inputs' :{
        'data': 'input_data',
        'bool': 'input_bool'
    },
    'output': 'output'
}

IMG_SIZE = 320
NUM_CHANNELS = 3
NUM_FEATURES = 2
EPOCHS = 1001
BATCH_SIZE = 36
LEARNING_RATE = 0.001
LOSS_MODE = 'mse'
MODEL_NAME = 'cnn_ep%s_bts%s_lr%s' % (EPOCHS, BATCH_SIZE, int(1/LEARNING_RATE))

# In here, the training.csv should be organized in a way that col: img_id, feature1, feature2, ....

DATA_FP = './data/generic_data'
DATA_FN_TRAIN = '%s/train.csv' % DATA_FP
IMG_ROOT_DIR = '%s/imgs' % DATA_FP
IMG_ID_NAME = 'image_id'