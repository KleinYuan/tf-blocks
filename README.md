# Announcement

To further increase the scope of this repo, the original `KleinYuan/cnn` repo has been deprecated and
saved in this [branch](https://github.com/KleinYuan/tf-blocks/tree/master-deprecated).

The scope of this branch has been extended to `tf-blocks` since Jul., 2018.

# 1. TF-Blocks

The scope of this repo is to provide a modularized architecture of tensorflow project so that
you can put more focus on each individual part more independently.


### 2.1 Motivation

Since tensorflow is a Symbolic Programming framework, quite different than Pytorch or Caffe2,
the computation, neural network architecture and training are totally four independent things!

Therefore I separate them into following four parts and also provide template module for each of them, 
as well as different other implementations such as CNN, RNN, ... etc. so that you can refer to to develop
your own ones. That's why I cal this repo `tf-blocks`.

I used this repo as basis for my actual daily researches to speed up prototyping and try my best to put everything
in a way of production ready quality.

**If you like it, please star it!**

**If you wanna contribute, feel free to submit Pull Request to make it better!**


# 2. Components

### 2.1 Graph Computation
 
I separate the computation part and placeholder the actual network architecture  so that when you need to investigate/develop
this part, you don't need to mix the improvements on graph computation with the network.


### 2.2 Network

Network is all about architecture and you may need to follow some constraints so that the graph module can be compatible
with this module. However, it's constructed in a way that you truly can just focus on the architecture.

### 2.3 Training

While you have network/graph computation done, all left is how you wanna train it, including:

* Which device you wanna run it? (CPU/GPU/Multi-Towers)

* How you wanna manage the data feed in/batch?

* How to config the tensorboard summary display?


### 2.4 DataGenerator

This part is all about data:

* How to load data from file path?

* How to pre-process the data?

* How to set the ratio of data?


# 3. Run Example App

The scope of this repo is to make it way easier to build a customized training app which you can focus
on separate part.

For demo cases, I made a CNN base neural network pipeline with MNIST dataset API (cuz it's just free).

The demo app is composed of:

```
/apps/cnn_training_app.py
/config/cnn_config.py
/models/cnn_net.py
```

Basically, you can run it with:

```
# Navigate to /tf-blocks/

python apps/cnn_training_app.py

```

# 4. How to develop your own training app?

0. If I am lazy

Just look at how those two scripts do and mimic them:

```
/apps/base_training_app.py
/models/base_net.py
```

1. Create a Model

```
touch /models/${fancy_name}_net.py
```

Then create a class inherent from `core.base_net.BaseNet`, finish the network architecture wrapped up in
`define_net` function.

Notes:

* You need to update `x_pl/y_pl/is_training` three tf placeholders.

2. Create a training app

```
touch /apps/${fancy_name}_training_app.py
```

Then create a class inherent from `core.base_data_generator.BaseDataGenerator`, finish the data loader with your own setup wrapped up in
`load_data` function.

Last, add the code below:

```

def main():
	# Config logger
	tf.logging.set_verbosity(tf.logging.INFO)
	logger = tf.logging
	# Initialize Four Modules: Data, Trainer, Net, Graph
	data_generator = ${fancy_name}DataGenerator()
	net = Net(config=config, logger=logger)
	graph_model = BaseGraph(net=net, config=config, logger=logger)
	trainer = BaseTrainer(graph_model=graph_model, config=config, logger=logger)
	# Run Training
	trainer.train(data_generator=data_generator)


if __name__ == "__main__":
	main()
```

Notes:

* You need to update `train_data/val_data/test_data`, each of which is a dictionary with `x` (data) and `y` (label)

* Create a config file if you need under `/config/${fancy_name}_config.py`

3. Run Training!

```
python apps/${fancy_name}_training_app.py
```

# 5. Examples

**Architecture**             |  **Graph**
:-------------------------:|:-------------------------:
CNN| ![CNN](https://user-images.githubusercontent.com/8921629/42332548-1c64c040-802d-11e8-8ab3-14ea4758099c.png)
RNN | ![RNN](https://user-images.githubusercontent.com/8921629/42332535-145d11d6-802d-11e8-880b-891cabbd63a5.png)
GAN | TBA

# 6. Todo Items

- [X] Adding RNN example

- [ ] Adding GAN example

- [ ] Adding a generic inference app