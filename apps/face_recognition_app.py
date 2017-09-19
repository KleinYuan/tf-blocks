import os
import numpy as np
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

image_size = 96
num_channels = 1
num_keypoints = 30
epochs = 1001
batch_size = 36
loss_mode = 'mse'

data_fp = '../data/kaggle_face'
data_fn_train = '%s/training.csv' % data_fp
data_fn_test = '%s/test.csv' % data_fp


class FaceDataSets(DataSets):
    # Download the data from https://www.kaggle.com/c/facial-keypoints-detection
    # Then unzip the zips and put IdLookUpTable.csv/training.csv/test.csv under data/kaggleFace

    def _load(self, mode, fn):
        print 'Loading from fp : %s' % fn

        df = read_csv(os.path.expanduser(fn))
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
        df = df.dropna()

        X = np.vstack(df['Image'].values) / 255.
        X = X.astype(np.float32)

        if mode == 'train':
            y = df[df.columns[:-1]].values
            y = (y - 48) / 48
            X, y = shuffle(X, y, random_state=42)
            y = y.astype(np.float32)
        else:
            y = None

        X = X.reshape(-1, image_size, image_size, num_channels)

        self.x_data = X
        self.y_data = y


def run():
    kaggelFaceDataSets = FaceDataSets(fp=data_fp)
    kaggelFaceDataSets.load(mode='train', fn=data_fn_train)
    # kaggelFaceDataSets.load(mode='test', fn=data_fn_train)

    x_train, x_test, y_train, y_test = train_test_split(kaggelFaceDataSets.x_train, kaggelFaceDataSets.y_train, test_size=0.3)
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

    net = Net(dropout=True,
              num_output=num_keypoints,
              num_conv=3,
              num_fc=2)

    graph_model = Graph(input_shape=[image_size, image_size, num_channels],
                        output_shape=[num_keypoints],
                        net=net)

    loss_calculator = lossCalculator(mode=loss_mode)

    trainer = Trainer(graph_model=graph_model,
                      epochs=epochs,
                      batch_size=batch_size,
                      logdir='%s/train' % data_fp,
                      save_path='%s/save' % data_fp)
    trainer.train(data=data,
                  loss_calculator=loss_calculator)


if __name__ == "__main__":
    run()
