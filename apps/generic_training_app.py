import os
import numpy as np
import cv2

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from pandas.io.parsers import read_csv
from models.data import DataSets
from models.cnn import Net
from models.graph import Graph
from models.loss import lossCalculator
from services.train_services import Trainer
from services.info_services import get_env_info

get_env_info()

IMG_SIZE = 320
NUM_CHANNELS = 3
NUM_FEATURES = 2
EPOCHS = 1001
BATCH_SIZE = 36
LEARNING_RATE = 0.001
LOSS_MODE = 'mse'
MODEL_NAME = 'cnn_ep%s_bts%s_lr%s' % (EPOCHS, BATCH_SIZE, LEARNING_RATE)

# In here, the training.csv should be organized in a way that col: img_id, feature1, feature2, ....

DATA_FP = './data/generic_data'
DATA_FN_TRAIN = '%s/train.csv' % DATA_FP
DATA_FN_TEST = '%s/test.csv' % DATA_FP

IMG_FORMAT = 'jpg'
IMG_ROOT_DIR = '%s/imgs' % DATA_FP
IMG_ID_NAME = 'image_id'


class GenericDataSets(DataSets):
    # Download the data from https://www.kaggle.com/c/facial-keypoints-detection
    # Then unzip the zips and put IdLookUpTable.csv/training.csv/test.csv under data/kaggleFace

    def _load(self, mode, fn):
        x_data = []
        y_data = []
        print 'Loading from fp : %s' % fn

        df = read_csv(os.path.expanduser(fn))
        print 'Found keys: ', df.keys()

        img_ids = df[IMG_ID_NAME].values
        print 'Loading imgs, resize and normalize it! '
        print img_ids
        for img_id in img_ids:
            img = cv2.imread('%s/%s.%s' % (IMG_ROOT_DIR, img_id, IMG_FORMAT))
            if img is None:
                print 'Found None img: %s' % img_id
                raise
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            x_data.append(img)

        x_data = np.array(x_data)
        x_data = x_data.astype(np.float32)

        if mode == 'train':
            print 'Loading features and normalize them!'

            y_data = df[df.columns[1:NUM_FEATURES+1]].values
            y_data = y_data.astype(np.float32)

            # print 'Shuffling....'
            x_data, y_data = shuffle(x_data, y_data, random_state=42)
        else:
            y_data = None

        print 'Assign to instance object.'
        x_data = x_data.reshape(-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
        self.x_data = x_data
        self.y_data = y_data


def run():
    data_sets = GenericDataSets(fp=DATA_FP)
    data_sets.load(mode='train', fn=DATA_FN_TRAIN)

    x_train, x_test, y_train, y_test = train_test_split(data_sets.x_train, data_sets.y_train, test_size=0.3)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
    data = {
        'train': {
            'x': x_train,
            'y': y_train
        },
        'val': {
            'x': x_val,
            'y': y_val
        },
        'test': {
            'x': x_test,
            'y': y_test
        }
    }

    cnn_config = {
        'conv_kernel_config': {
            'size': 3,
            'depth_init': 32
        },
        'pool_size':2,
        'dropout_config':{
            'keep_prob_init': 0.9,
            'decay': 0.1
        },
        'fc_relu_config': {
            'size': 1000
        }
    }
    net = Net(dropout=True,
              num_output=NUM_FEATURES,
              num_conv=3,
              num_fc=2,
              cnn_config=cnn_config)

    graph_model = Graph(input_shape=[IMG_SIZE, IMG_SIZE, NUM_CHANNELS],
                        output_shape=[NUM_FEATURES],
                        net=net,
                        learning_rate=LEARNING_RATE)

    loss_calculator = lossCalculator(mode=LOSS_MODE)

    trainer = Trainer(graph_model=graph_model,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      logdir='%s/train' % DATA_FP,
                      save_path='%s/save/%s' % (DATA_FP, MODEL_NAME))
    trainer.train(data=data,
                  loss_calculator=loss_calculator)


if __name__ == "__main__":
    run()
