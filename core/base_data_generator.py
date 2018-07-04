class BaseDataGenerator(object):

	train_data = None
	val_data = None
	test_data = None

	def __init__(self, *args, **kwargs):
		self.load_data()

	def load_data(self, *args, **kwargs):
		"""
		1. Loading data from path via diverse formats
		2. Update ${type}_data so that it contains ${type}['x'], ${type}['y']
		"""
		raise NotImplementedError

	@staticmethod
	def batch_iterator(xs, ys, batch_size=1):
		l = len(xs)
		assert l == len(ys)
		for ndx in range(0, l, batch_size):
			yield xs[ndx:min(ndx + batch_size, l)], ys[ndx:min(ndx + batch_size, l)]

	@property
	def train(self):
		return self.train_data['x'], self.train_data['y']

	@property
	def val(self):
		return self.val_data['x'], self.val_data['y']

	@property
	def test(self):
		return self.test_data['x'], self.test_data['y']
