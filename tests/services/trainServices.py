import unittest
from services.trainServices import train


class TestTrainService(unittest.TestCase):
    def setUp(self):
        self.graph = 'mock'
        self.epoch = 1001
        self.batch_size = 36
        self.feed_dict_succ = {
            'x': 'mock',
            'y': 'mock',
            'training': 'mock'
        }
        self.feed_dict_fail = {
            'x': 'mock',
            'y': 'mock'
        }
        self.data_sucss = {
            'train': {
                'x': 'mock',
                'y': 'mock'
            },
            'val': {
                'x': 'mock',
                'y': 'mock'
            },
            'test': {
                'x': 'mock',
                'y': 'mock'
            }

        }
        self.data_fail = {
            'train': {
                'x': 'mock',
                'y': 'mock'
            },
            'val': {
                'x': 'mock',
                'y': 'mock'
            }
        }

        self.optimizer = 'mock'

        self.loss_calculator = MockLossCalculator()

        self.logdir = 'mock'
        self.save_path = './mock'

    def test_train(self):
        trained_succ = train(graph=self.graph,
                             epochs=self.epoch,
                             batch_size=self.batch_size,
                             feed_dict=self.feed_dict_succ,
                             data=self.data_sucss,
                             optimizer=self.optimizer,
                             loss_calculator=self.loss_calculator,
                             logdir=self.logdir,
                             save_path=self.save_path,
                             unittest=True)

        self.assertTrue(trained_succ)


class MockLossCalculator:
    def __init__(self):
        print 'Mocking Loss Calculator'

    def calculate(self, x, y):
        print x - y

if __name__ == '__main__':
    unittest.main()
