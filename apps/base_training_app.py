import tensorflow as tf
from core.base_net import BaseNet
from core.base_graph import BaseGraph
from core.base_trainer import BaseTrainer
from core.base_data_generator import BaseDataGenerator
from config import base_config as config


class MyDataGenerator(BaseDataGenerator):
    def load_data(self):
        """
        Put your data loading code here!
        """
        return


def main():
    """
    This is a base training app.
    In most cases, you don't need to modify any lines of this script.
    """
    # Config logger
    tf.logging.set_verbosity(tf.logging.INFO)
    logger = tf.logging
    # Initialize Four Modules: Data, Trainer, Net, Graph
    data_generator = MyDataGenerator()
    net = BaseNet(config=config, logger=logger)
    graph_model = BaseGraph(net=net, config=config, logger=logger)
    trainer = BaseTrainer(graph_model=graph_model, config=config, logger=logger)
    # Run Training
    trainer.train(data_generator=data_generator)


if __name__ == "__main__":
    main()
