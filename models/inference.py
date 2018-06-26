import cv2
import numpy as np
import tensorflow as tf
from config import config


class Net:

    def __init__(self, graph_fp):
        self.graph_fp = graph_fp
        self.graph = None
        self.input_data_tensor = None
        self.input_bool_tensor = None
        self.output_tensor = None
        self.prediction = None

        self.in_progress = False
        self.session = None
        with tf.device('/gpu:0'):
            self._load_graph()
            self._init_predictor()

    def _load_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_fp, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        tf.get_default_graph().finalize()

    def _init_predictor(self):
        tf_config = tf.ConfigProto(device_count={'gpu': 0}, log_device_placement=True)
        tf_config.gpu_options.allow_growth = True
        with self.graph.as_default():
            self.session = tf.Session(config=tf_config, graph=self.graph)
            self.input_data_tensor = self.graph.get_tensor_by_name('%s:0' % config.NODE_NAMES['inputs']['data'])
            self.input_bool_tensor = self.graph.get_tensor_by_name('%s:0' % config.NODE_NAMES['inputs']['bool'])
            self.output_tensor = self.graph.get_tensor_by_name('out/add:0')

    def predict(self, img):
        self.in_progress = True
        self.prediction = None
        with self.graph.as_default():
            print '[INFO] Read the image ..'

            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            x_data = np.array([img])
            x_data = x_data.astype(np.float32)
            x_data = x_data.reshape(-1, config.IMG_SIZE, config.IMG_SIZE, config.NUM_CHANNELS)

            print '[INFO] Detecting objects ...'
            [prediction] = self.session.run(
                [self.output_tensor],
                feed_dict={
                    self.input_data_tensor: x_data,
                    self.input_bool_tensor: False
                })
            print '[Info] Prediction is : ', prediction
            self.prediction = prediction[0]
        self.in_progress = False

    def get_prediction(self):
        return self.prediction

    def get_status(self):
        return self.in_progress

    def kill_predictor(self):
        self.session.close()
        self.session = None
        self.prediction = None
