from copy import deepcopy


class DataSets:

    # In here, we wanna support multiple ways of organizing datasets, like raws images, hd5, tfrecords, ...
    # We only provide a model here and eventually,
    # you wanna override most functions in this model with your own loader ...

    def __init__(self, fp):
        print 'Initializing a dataset ...'
        self.fp = fp
        self.x_data = None
        self.y_data = None

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def load(self, mode, fn):
        print 'Loading data ...'
        self._load(mode=mode, fn=fn)
        if mode == 'train':
            self.x_train = deepcopy(self.x_data)
            self.y_train = deepcopy(self.y_data)
        elif mode == 'val':
            self.x_val = deepcopy(self.x_data)
            self.y_val = deepcopy(self.y_data)
        elif mode == 'test':
            self.x_test = deepcopy(self.x_data)
            self.y_test = deepcopy(self.y_data)

        print 'Data (%s) loaded !' % mode

    def _load(self, mode, fn):
        print 'To be overrided!'
