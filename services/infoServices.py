import tensorflow as tf
import multiprocessing
from tensorflow.python.client import device_lib


def get_env_info():
    local_device_protos = device_lib.list_local_devices()
    print 'Tensorflow: %s' % tf.__version__
    print 'CPU number: %s' % multiprocessing.cpu_count()
    print 'Computation unit available: %s' % len(local_device_protos)
    print 'GPU available: %s' % len([x.name for x in local_device_protos if x.device_type == 'GPU'])
