class BaseNet(object):
    """
    Arguably, the best practice in general should be, you shall always define a placeholder to indicate
    whether the graph is doing training or evaluating so that you can dynamically do something like changing
    drop-out rate.
    Therefore, the constraint is also:
        - You shall always define a is_training_pl (pl is for placeholder) to indicate whether you are
          doing training or not

    Sometimes, you may need more than one x_pl for example you are doing multi-task architecture, then you wanna
    do some tweaks on your own (I assume if you go that far, you may be familiar enough with tensorflow yo bro?).
    I try not to wrap up the x_pl into a list so that it's hard to track the placeholder names. But we will see how
    far we can go from here.
    """
    x_pl = None
    y_pl = None
    is_training_pl = None

    def __init__(self, config, logger, *args, **kwargs):
        self.logger = logger
        self.config = config
        self.logger.info('Initializing network...')

    def define_net(self, *args, **kwargs):
        """
        Put your neural network architecture here

        :return: prediction
        """
        self.logger.info('Defining network...')
        raise NotImplementedError

    def get_placeholders(self):
        return self.x_pl, self.y_pl, self.is_training_pl
